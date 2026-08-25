/**
 * Transporte Web Bluetooth: la app contra el dispositivo de verdad.
 *
 * Sustituye a `MockTransport`, que reproduce la física pero se la inventa. La
 * interfaz es la misma —`TelemetryTransport`— así que enchufarlo es cambiar qué
 * clase se instancia, no reescribir la app. Para eso estaba la interfaz.
 *
 * Formato de los datos
 * --------------------
 * Binario empaquetado little-endian, el mismo que escribe
 * `firmware/arduino/pneumocoach_capture/ble.h`. Ese fichero es la otra mitad de
 * este contrato y los dos hay que cambiarlos juntos.
 *
 * Dónde funciona
 * --------------
 * Web Bluetooth existe en Chrome y Edge; no en Safari ni en Firefox. En Android
 * basta con Chrome. En iOS no hay forma, y por eso `MockTransport` sigue en el
 * árbol: la app tiene que poder demostrarse sin dispositivo.
 *
 * Y exige contexto seguro: HTTPS, o `localhost`. Servido por HTTP desde una IP
 * de la red local, el navegador ni siquiera muestra el selector de dispositivos
 * — falla de una forma que parece un problema de Bluetooth y es de origen.
 */

import {
  CHAR_SENSORS,
  CHAR_STATUS,
  CHAR_VERDICT,
  PNEUMO_SERVICE,
  type TelemetryTransport,
} from "./ble";
import {
  CONFIDENCE_FLOOR,
  type DeviceStatus,
  type Reading,
  type SensorFrame,
  type VerdictKey,
} from "./telemetry";

/**
 * Orden de clases del firmware, que es el del modelo entrenado.
 *
 * `artifact` ya no es una clase: entrenarla junto a las técnicas costaba 5
 * puntos en la pregunta que importa. El índice 255 no es un error sino el
 * dispositivo negándose a adivinar por debajo del piso de confianza, y la app
 * tiene que poder distinguir eso de «todavía no ha llegado nada».
 */
const CLASES: VerdictKey[] = ["diaphragmatic", "thoracic", "rapid_shallow"];

/** Longitud de la traza que se guarda para las gráficas. */
const TRAZA = 240;

type Suscriptor<T> = (v: T) => void;

function emisor<T>() {
  const subs = new Set<Suscriptor<T>>();
  return {
    on(cb: Suscriptor<T>) {
      subs.add(cb);
      return () => subs.delete(cb);
    },
    emit(v: T) {
      subs.forEach((s) => s(v));
    },
  };
}

export class WebBluetoothTransport implements TelemetryTransport {
  private device: BluetoothDevice | null = null;
  private server: BluetoothRemoteGATTServer | null = null;
  private readonly evEstado = emisor<DeviceStatus>();
  private readonly evLectura = emisor<Reading>();
  private readonly evTrama = emisor<SensorFrame>();

  private tilt: number[] = [];
  private axial: number[] = [];
  private pitch = 0;
  private gyroRms = 0;

  private estado: DeviceStatus = {
    state: "disconnected",
    name: "PneumoCoach",
    rssi: 0,
    battery: 0,
    dropped: 0,
    effectiveHz: 0,
    firmware: "",
    imu: "MPU6500",
  };

  onStatus(cb: Suscriptor<DeviceStatus>) {
    cb(this.estado);
    return this.evEstado.on(cb);
  }
  onReading(cb: Suscriptor<Reading>) {
    return this.evLectura.on(cb);
  }
  onFrame(cb: Suscriptor<SensorFrame>) {
    return this.evTrama.on(cb);
  }

  private publicaEstado(parcial: Partial<DeviceStatus>) {
    this.estado = { ...this.estado, ...parcial };
    this.evEstado.emit(this.estado);
  }

  async connect(): Promise<void> {
    if (!navigator.bluetooth) {
      throw new Error(
        "Este navegador no tiene Web Bluetooth. En Android usa Chrome; " +
          "en iOS no está disponible y hay que usar el modo demostración.",
      );
    }
    this.publicaEstado({ state: "connecting" });

    // El filtro por servicio evita mostrarle al usuario una lista con todos los
    // cacharros Bluetooth de la sala. En una demo eso importa.
    this.device = await navigator.bluetooth.requestDevice({
      filters: [{ services: [PNEUMO_SERVICE] }],
      optionalServices: [PNEUMO_SERVICE],
    });

    this.device.addEventListener("gattserverdisconnected", () => {
      this.publicaEstado({ state: "disconnected", effectiveHz: 0 });
    });

    this.server = (await this.device.gatt?.connect()) ?? null;
    if (!this.server) throw new Error("No se pudo abrir el GATT");

    const svc = await this.server.getPrimaryService(PNEUMO_SERVICE);
    const [ver, sen, est] = await Promise.all([
      svc.getCharacteristic(CHAR_VERDICT),
      svc.getCharacteristic(CHAR_SENSORS),
      svc.getCharacteristic(CHAR_STATUS),
    ]);

    ver.addEventListener("characteristicvaluechanged", (e) =>
      this.leeVeredicto((e.target as BluetoothRemoteGATTCharacteristic).value!),
    );
    sen.addEventListener("characteristicvaluechanged", (e) =>
      this.leeSensores((e.target as BluetoothRemoteGATTCharacteristic).value!),
    );
    est.addEventListener("characteristicvaluechanged", (e) =>
      this.leeEstado((e.target as BluetoothRemoteGATTCharacteristic).value!),
    );

    await Promise.all([
      ver.startNotifications(),
      sen.startNotifications(),
      est.startNotifications(),
    ]);

    this.publicaEstado({
      state: "connected",
      name: this.device.name ?? "PneumoCoach",
      firmware: "captura v1",
    });
  }

  async disconnect(): Promise<void> {
    this.server?.disconnect();
    this.server = null;
    this.device = null;
    this.publicaEstado({ state: "disconnected", effectiveHz: 0 });
  }

  /** 22 B · u8 clase, u8 confianza, 5 × f32 */
  private leeVeredicto(v: DataView) {
    if (v.byteLength < 22) return;
    const clase = v.getUint8(0);
    const conf = v.getUint8(1) / 255;
    const bpm = v.getFloat32(2, true);
    const ie = v.getFloat32(6, true);
    const tiltDeg = v.getFloat32(10, true);
    const axialMg = v.getFloat32(14, true);
    const ratio = v.getFloat32(18, true);

    // 255 es «me niego a adivinar», no un fallo de transmisión. Se representa
    // como `artifact`, que es la clase que la interfaz ya usa para no dar
    // consejo, con la confianza real para que se vea POR QUÉ se abstuvo.
    const verdict: VerdictKey =
      clase < CLASES.length ? CLASES[clase] : "artifact";

    this.evLectura.emit({
      t: Date.now(),
      verdict,
      confidence: conf,
      bpm,
      ieRatio: ie,
      tiltDeg,
      axialMg,
      ratio,
      hfRatio: 0,
    });
  }

  /** 16 B · 4 × f32 */
  private leeSensores(v: DataView) {
    if (v.byteLength < 16) return;
    const tilt = v.getFloat32(0, true);
    const axial = v.getFloat32(4, true);
    this.pitch = tilt;
    this.gyroRms = v.getFloat32(12, true);

    this.tilt = [...this.tilt, tilt].slice(-TRAZA);
    this.axial = [...this.axial, axial * 1000].slice(-TRAZA);

    this.evTrama.emit({
      tilt: this.tilt,
      axial: this.axial,
      pitch: this.pitch,
      gyroRms: this.gyroRms,
      // El BMP180 se quedó fuera de alcance: no se inventan lecturas. La
      // pestaña de sensores muestra estos campos como no disponibles.
      pressureHpa: 0,
      tempC: 0,
      altitudeM: 0,
    });
  }

  /** 14 B · u8 fase, u8 banderas, f32 restante, 2 × u16, f32 contraste */
  private leeEstado(v: DataView) {
    if (v.byteLength < 14) return;
    const fase = v.getUint8(0);
    const banderas = v.getUint8(1);
    this.publicaEstado({
      state: "connected",
      effectiveHz: 50,
      // Se reutiliza `firmware` para mostrar la fase de sesión en la barra
      // superior: es la información que de verdad quiere ver alguien que está
      // haciendo el ejercicio.
      firmware: FASES[fase] ?? `fase ${fase}`,
      dropped: 0,
      battery: (banderas & 2) === 2 ? 100 : 0,
    });
  }
}

/** Las mismas fases que muestra la pantalla OLED del dispositivo. */
export const FASES: Record<number, string> = {
  0: "Asentando",
  1: "Preparado",
  2: "Calibrando · abdomen",
  3: "Calibrando · pecho",
  4: "Calibrado",
  5: "Sesión activa",
};

export { CONFIDENCE_FLOOR };
