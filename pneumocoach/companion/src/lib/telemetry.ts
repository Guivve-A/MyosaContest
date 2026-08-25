/**
 * Contrato de telemetría del dispositivo PneumoCoach.
 *
 * Estos tipos no son decorativos: reflejan lo que el firmware realmente emite.
 * Las cuatro clases, la ventana de 12 s con salto de 3 s, el piso de confianza
 * de 0.60 y las 29 características salen de `ml/pneumocoach/config.py`, que es
 * la fuente única de verdad compartida entre el modelo y el ESP32.
 *
 * Si `config.py` cambia, este archivo cambia. No al revés.
 */

export const FS_HZ = 50;
export const WINDOW_S = 12;
export const HOP_S = 3;
export const CONFIDENCE_FLOOR = 0.6;
export const N_FEATURES = 29;

export type VerdictKey =
  | "diaphragmatic"
  | "thoracic"
  | "rapid_shallow"
  | "artifact";

export interface VerdictMeta {
  key: VerdictKey;
  label: string;
  short: string;
  /** Qué significa clínicamente, en el idioma del paciente y no del ingeniero. */
  meaning: string;
  /** Es la técnica que el coaching persigue. */
  coachedOk: boolean;
  tone: "ok" | "warn" | "alert" | "mute";
}

export const VERDICTS: Record<VerdictKey, VerdictMeta> = {
  diaphragmatic: {
    key: "diaphragmatic",
    label: "Diafragmática",
    short: "Diafragm.",
    meaning: "El abdomen se expande y el pecho superior permanece quieto.",
    coachedOk: true,
    tone: "ok",
  },
  thoracic: {
    key: "thoracic",
    label: "Torácica",
    short: "Torácica",
    meaning: "El pecho superior se eleva. Respiración superficial.",
    coachedOk: false,
    tone: "warn",
  },
  rapid_shallow: {
    key: "rapid_shallow",
    label: "Rápida superficial",
    short: "Rápida",
    meaning: "Frecuencia alta con poco volumen por respiración.",
    coachedOk: false,
    tone: "alert",
  },
  artifact: {
    key: "artifact",
    label: "Señal alterada",
    short: "Alterada",
    meaning: "Tos, habla o movimiento. El dispositivo se abstiene de evaluar.",
    coachedOk: false,
    tone: "mute",
  },
};

export const VERDICT_ORDER: VerdictKey[] = [
  "diaphragmatic",
  "thoracic",
  "rapid_shallow",
  "artifact",
];

/** Una lectura por ventana, emitida cada HOP_S segundos. */
export interface Reading {
  t: number;
  verdict: VerdictKey;
  confidence: number;
  /** Respiraciones por minuto. */
  bpm: number;
  /** Relación inspiración:espiración, expresada como 1:N. */
  ieRatio: number;
  /** Amplitud del canal de inclinación, en grados. */
  tiltDeg: number;
  /** Amplitud del canal axial, en miligravedades. */
  axialMg: number;
  /** log10(tilt/axial). Discriminador previsto en el diseño original; medido
   *  en d = -0.05 sobre un torax real y por tanto NO usado para clasificar.
   *  Se sigue mostrando en la pestana de sensores por transparencia. */
  ratio: number;
  /** Fracción de energía sobre 2 Hz: delata movimiento o habla. */
  hfRatio: number;
}

export type ConnectionState =
  | "disconnected"
  | "scanning"
  | "connecting"
  | "connected";

export interface DeviceStatus {
  state: ConnectionState;
  name: string;
  rssi: number;
  battery: number;
  /** Muestras perdidas desde el inicio de la sesión. */
  dropped: number;
  /** Tasa de muestreo efectiva medida en el dispositivo. */
  effectiveHz: number;
  firmware: string;
  imu: string;
}

export interface SensorFrame {
  /** Últimas muestras del canal de inclinación, ya filtradas (grados). */
  tilt: number[];
  /** Últimas muestras del canal axial, ya filtradas (mg). */
  axial: number[];
  /** Ángulo de cabeceo crudo, antes del paso banda (grados). */
  pitch: number;
  /** Magnitud del giroscopio, para detectar movimiento (dps). */
  gyroRms: number;
  /** Presión barométrica del BMP180 (hPa). */
  pressureHpa: number;
  /** Temperatura del BMP180 (°C). */
  tempC: number;
  /** Altitud relativa derivada de la presión (m). */
  altitudeM: number;
}

/** Estadísticas acumuladas de la sesión en curso. */
export interface SessionStats {
  startedAt: number;
  elapsedS: number;
  /** Fracción del tiempo evaluable pasado en técnica correcta. */
  onTechnique: number;
  meanBpm: number;
  meanIe: number;
  bestStreakS: number;
  breaths: number;
  /** Ventanas descartadas por señal alterada. */
  discarded: number;
  perVerdict: Record<VerdictKey, number>;
}

export function verdictColorVar(tone: VerdictMeta["tone"]): string {
  return `var(--verdict-${tone})`;
}

export function formatIe(ratio: number): string {
  return `1:${ratio.toFixed(1)}`;
}

export function formatClock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
