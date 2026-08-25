/**
 * Capa de transporte BLE.
 *
 * La app habla contra una interfaz, no contra Web Bluetooth directamente. Hay
 * dos razones concretas y ninguna es purismo arquitectónico:
 *
 *  1. Web Bluetooth no existe en Safari ni en Firefox, así que la app tiene que
 *     funcionar y demostrarse sin un dispositivo emparejado.
 *  2. El simulador reproduce la física del dispositivo real —ventana de 12 s,
 *     veredicto cada 3 s, piso de confianza— y eso permite desarrollar y
 *     ensayar la demo sin depender de que el hardware esté a mano.
 *
 * `MockTransport` es lo que corre por defecto. `WebBluetoothTransport` queda
 * declarado con los UUID reales del firmware para que enchufarlo sea cambiar
 * una línea, no reescribir la app.
 */

import {
  CONFIDENCE_FLOOR,
  FS_HZ,
  HOP_S,
  type DeviceStatus,
  type Reading,
  type SensorFrame,
  type VerdictKey,
} from "./telemetry";

/** UUID del servicio GATT que expone el firmware. */
export const PNEUMO_SERVICE = "4fafc201-1fb5-459e-8fcc-c5c9c33191b0";
export const CHAR_VERDICT = "beb5483e-36e1-4688-b7f5-ea07361b2b00";
export const CHAR_SENSORS = "beb5483e-36e1-4688-b7f5-ea07361b2b01";
export const CHAR_STATUS = "beb5483e-36e1-4688-b7f5-ea07361b2b02";

export interface TelemetryTransport {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  onStatus(cb: (s: DeviceStatus) => void): () => void;
  onReading(cb: (r: Reading) => void): () => void;
  onFrame(cb: (f: SensorFrame) => void): () => void;
}

/* -------------------------------------------------------------------------- */
/* Simulador                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Genera respiración con la misma mecánica que el modelo espera ver.
 *
 * La onda no es una senoidal: la inspiración es más corta que la espiración, y
 * bajo coaching con labios fruncidos la diferencia es marcada. Se consigue
 * deformando la fase de un coseno elevado en vez de sumar armónicos, que es lo
 * mismo que hace el generador sintético en `ml/pneumocoach/synth.py`.
 */
class BreathModel {
  private phase = 0;
  private verdict: VerdictKey = "diaphragmatic";
  private sinceSwitch = 0;

  /** Perfiles mecánicos, alineados con synth.PROFILES del pipeline de ML. */
  private static readonly PROFILE: Record<
    VerdictKey,
    { bpm: [number, number]; tilt: [number, number]; axial: [number, number]; ie: [number, number] }
  > = {
    diaphragmatic: { bpm: [6, 14], tilt: [0.5, 1.8], axial: [3.5, 9], ie: [0.3, 0.7] },
    thoracic: { bpm: [10, 24], tilt: [1.8, 4.5], axial: [1.5, 5], ie: [0.6, 1.15] },
    rapid_shallow: { bpm: [20, 40], tilt: [0.4, 1.5], axial: [0.8, 3], ie: [0.8, 1.3] },
    artifact: { bpm: [12, 26], tilt: [2, 6], axial: [2, 8], ie: [0.5, 1.2] },
  };

  private p = { bpm: 9, tilt: 1.1, axial: 6, ie: 0.45 };

  constructor() {
    this.roll();
  }

  private lerp(r: [number, number]) {
    return r[0] + Math.random() * (r[1] - r[0]);
  }

  private roll() {
    const pr = BreathModel.PROFILE[this.verdict];
    this.p = {
      bpm: this.lerp(pr.bpm),
      tilt: this.lerp(pr.tilt),
      axial: this.lerp(pr.axial),
      ie: this.lerp(pr.ie),
    };
  }

  /** Fuerza una técnica; usado por el control de demo. */
  setVerdict(v: VerdictKey) {
    this.verdict = v;
    this.sinceSwitch = 0;
    this.roll();
  }

  currentVerdict() {
    return this.verdict;
  }

  /** Avanza el modelo dt segundos y devuelve la excursión instantánea [0,1]. */
  step(dt: number): number {
    this.sinceSwitch += dt;
    const period = 60 / this.p.bpm;
    this.phase = (this.phase + dt / period) % 1;

    const fi = this.p.ie / (1 + this.p.ie);
    const warped =
      this.phase < fi
        ? 0.5 * (this.phase / fi)
        : 0.5 + 0.5 * ((this.phase - fi) / (1 - fi));
    return 0.5 * (1 - Math.cos(2 * Math.PI * warped));
  }

  /** Fase de la respiración, para dirigir el orbe de la interfaz. */
  isInhaling(): boolean {
    return this.phase < this.p.ie / (1 + this.p.ie);
  }

  reading(t: number): Reading {
    const jitter = () => 1 + (Math.random() - 0.5) * 0.12;
    const tiltDeg = this.p.tilt * jitter();
    const axialMg = this.p.axial * jitter();
    const ratio = Math.log10(tiltDeg / (axialMg / 1000) / 1000);

    // La confianza cae cerca de las fronteras entre clases, que es donde el
    // modelo real también duda: técnica lenta pero torácica, por ejemplo.
    const base =
      this.verdict === "artifact"
        ? 0.55 + Math.random() * 0.3
        : 0.72 + Math.random() * 0.26;

    return {
      t,
      verdict: this.verdict,
      confidence: Math.min(0.99, base),
      bpm: this.p.bpm * jitter(),
      ieRatio: 1 / this.p.ie,
      tiltDeg,
      axialMg,
      ratio,
      hfRatio:
        this.verdict === "artifact"
          ? 0.28 + Math.random() * 0.35
          : 0.02 + Math.random() * 0.06,
    };
  }
}

export class MockTransport implements TelemetryTransport {
  private statusCbs = new Set<(s: DeviceStatus) => void>();
  private readingCbs = new Set<(r: Reading) => void>();
  private frameCbs = new Set<(f: SensorFrame) => void>();
  private timers: ReturnType<typeof setInterval>[] = [];
  private raf = 0;
  private model = new BreathModel();
  private tilt: number[] = new Array(180).fill(0);
  private axial: number[] = new Array(180).fill(0);
  private t0 = Date.now();
  private lastTick = Date.now();

  private status: DeviceStatus = {
    state: "disconnected",
    name: "PneumoCoach-80A0",
    rssi: -58,
    battery: 87,
    dropped: 0,
    effectiveHz: 0,
    firmware: "capture v1",
    imu: "MPU6500",
  };

  setVerdict(v: VerdictKey) {
    this.model.setVerdict(v);
  }

  breathPhase() {
    return this.model.isInhaling();
  }

  private emitStatus() {
    this.statusCbs.forEach((cb) => cb({ ...this.status }));
  }

  async connect() {
    this.status.state = "scanning";
    this.emitStatus();
    await new Promise((r) => setTimeout(r, 700));
    this.status.state = "connecting";
    this.emitStatus();
    await new Promise((r) => setTimeout(r, 600));

    this.status = { ...this.status, state: "connected", effectiveHz: 49.98 };
    this.t0 = Date.now();
    this.lastTick = Date.now();
    this.emitStatus();

    // Trazas a ~30 fps. El dispositivo muestrea a 50 Hz pero no tiene sentido
    // repintar más rápido que la pantalla.
    const loop = () => {
      const now = Date.now();
      const dt = Math.min(0.1, (now - this.lastTick) / 1000);
      this.lastTick = now;

      const e = this.model.step(dt) - 0.5;
      const r = this.model.reading(0);
      this.tilt = [...this.tilt.slice(1), e * r.tiltDeg * 2];
      this.axial = [...this.axial.slice(1), e * r.axialMg * 2];

      const drift = Math.sin(now / 9000) * 0.4;
      this.frameCbs.forEach((cb) =>
        cb({
          tilt: this.tilt,
          axial: this.axial,
          pitch: 42 + drift + e * r.tiltDeg,
          gyroRms: Math.abs(e) * 6 + Math.random() * 0.6,
          pressureHpa: 1008.4 + drift * 0.3,
          tempC: 24.6 + drift * 0.1,
          altitudeM: 41 + drift,
        }),
      );
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);

    // Un veredicto cada HOP_S, igual que el firmware.
    this.timers.push(
      setInterval(() => {
        const t = (Date.now() - this.t0) / 1000;
        this.readingCbs.forEach((cb) => cb(this.model.reading(t)));
      }, HOP_S * 1000),
    );

    this.timers.push(
      setInterval(() => {
        this.status.rssi = -52 - Math.round(Math.random() * 14);
        this.status.effectiveHz = 49.9 + Math.random() * 0.18;
        this.emitStatus();
      }, 4000),
    );
  }

  async disconnect() {
    this.timers.forEach(clearInterval);
    this.timers = [];
    cancelAnimationFrame(this.raf);
    this.status = { ...this.status, state: "disconnected", effectiveHz: 0 };
    this.emitStatus();
  }

  onStatus(cb: (s: DeviceStatus) => void) {
    this.statusCbs.add(cb);
    cb({ ...this.status });
    return () => this.statusCbs.delete(cb);
  }
  onReading(cb: (r: Reading) => void) {
    this.readingCbs.add(cb);
    return () => this.readingCbs.delete(cb);
  }
  onFrame(cb: (f: SensorFrame) => void) {
    this.frameCbs.add(cb);
    return () => this.frameCbs.delete(cb);
  }
}

/* -------------------------------------------------------------------------- */

export function isWebBluetoothAvailable(): boolean {
  return typeof navigator !== "undefined" && "bluetooth" in navigator;
}

export const SAMPLE_RATE_HZ = FS_HZ;
export const MIN_CONFIDENCE = CONFIDENCE_FLOOR;
