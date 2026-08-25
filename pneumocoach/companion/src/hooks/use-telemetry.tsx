"use client";

/**
 * Estado de sesión compartido por las tres pestañas.
 *
 * Un solo contexto sostiene la conexión, el historial de veredictos y las
 * estadísticas acumuladas. Si cada pestaña abriera su propia suscripción, al
 * cambiar de pestaña se perdería la sesión — que es justo lo que un paciente
 * a mitad de un ejercicio de respiración no puede permitirse.
 */

import * as React from "react";

import { MockTransport } from "@/lib/ble";
import { WebBluetoothTransport } from "@/lib/ble-real";
import {
  HOP_S,
  VERDICT_ORDER,
  type DeviceStatus,
  type Reading,
  type SensorFrame,
  type SessionStats,
  type VerdictKey,
} from "@/lib/telemetry";

const HISTORY = 80;

interface TelemetryValue {
  status: DeviceStatus;
  reading: Reading | null;
  frame: SensorFrame | null;
  history: Reading[];
  stats: SessionStats;
  inhaling: boolean;
  connect: () => void;
  disconnect: () => void;
  /** Control de demostración: fuerza la técnica que simula el dispositivo. */
  forceVerdict: (v: VerdictKey) => void;
  /** True si los datos vienen del dispositivo y no del simulador. */
  modoReal: boolean;
  /** True si este navegador tiene Web Bluetooth. */
  hayBluetooth: boolean;
  usarDispositivoReal: (v: boolean) => void;
}

const Ctx = React.createContext<TelemetryValue | null>(null);

function emptyStats(): SessionStats {
  return {
    startedAt: Date.now(),
    elapsedS: 0,
    onTechnique: 0,
    meanBpm: 0,
    meanIe: 0,
    bestStreakS: 0,
    breaths: 0,
    discarded: 0,
    perVerdict: {
      diaphragmatic: 0,
      thoracic: 0,
      rapid_shallow: 0,
      artifact: 0,
    },
  };
}

export function TelemetryProvider({ children }: { children: React.ReactNode }) {
  /* Dispositivo real si el navegador puede; simulador si no.
   *
   * No es un apano: Web Bluetooth no existe en Safari ni en Firefox, y la app
   * tiene que poder ensenarse sin hardware. Lo que NO se hace es fingir: el
   * simulador se declara en la interfaz para que nadie confunda una demo con
   * una medicion. */
  const [modoReal, setModoReal] = React.useState(false);
  const transport = React.useMemo(
    () => (modoReal ? new WebBluetoothTransport() : new MockTransport()),
    [modoReal],
  );
  const hayBluetooth =
    typeof navigator !== "undefined" && "bluetooth" in navigator;
  const [status, setStatus] = React.useState<DeviceStatus>({
    state: "disconnected",
    name: "PneumoCoach-80A0",
    rssi: -58,
    battery: 87,
    dropped: 0,
    effectiveHz: 0,
    firmware: "capture v1",
    imu: "MPU6500",
  });
  const [reading, setReading] = React.useState<Reading | null>(null);
  const [frame, setFrame] = React.useState<SensorFrame | null>(null);
  const [history, setHistory] = React.useState<Reading[]>([]);
  const [stats, setStats] = React.useState<SessionStats>(emptyStats);
  const [inhaling, setInhaling] = React.useState(true);

  const streakRef = React.useRef(0);

  React.useEffect(() => {
    const offS = transport.onStatus(setStatus);
    const offF = transport.onFrame(setFrame);
    const offR = transport.onReading((r) => {
      setReading(r);
      setHistory((h) => [...h.slice(-(HISTORY - 1)), r]);

      setStats((prev) => {
        const perVerdict = { ...prev.perVerdict };
        perVerdict[r.verdict] += HOP_S;

        const evaluable = VERDICT_ORDER.filter((k) => k !== "artifact").reduce(
          (a, k) => a + perVerdict[k],
          0,
        );
        const ok = perVerdict.diaphragmatic;

        if (r.verdict === "diaphragmatic") streakRef.current += HOP_S;
        else streakRef.current = 0;

        const elapsed = (Date.now() - prev.startedAt) / 1000;
        const n = prev.breaths + Math.max(1, Math.round((r.bpm / 60) * HOP_S));

        return {
          ...prev,
          elapsedS: elapsed,
          perVerdict,
          onTechnique: evaluable > 0 ? ok / evaluable : 0,
          meanBpm: prev.meanBpm === 0 ? r.bpm : prev.meanBpm * 0.85 + r.bpm * 0.15,
          meanIe:
            prev.meanIe === 0 ? r.ieRatio : prev.meanIe * 0.85 + r.ieRatio * 0.15,
          bestStreakS: Math.max(prev.bestStreakS, streakRef.current),
          breaths: n,
          discarded: perVerdict.artifact / HOP_S,
        };
      });
    });
    return () => {
      offS();
      offF();
      offR();
    };
  }, [transport]);

  // El orbe respiratorio necesita la fase, que cambia mucho más rápido que los
  // veredictos. Se sondea aparte para no re-renderizar todo el árbol a 30 fps.
  React.useEffect(() => {
    if (status.state !== "connected") return;
    const id = setInterval(() => setInhaling("breathPhase" in transport
        ? (transport as MockTransport).breathPhase()
        : false), 80);
    return () => clearInterval(id);
  }, [transport, status.state]);

  /* Cambia entre dispositivo real y simulador. Se expone porque durante una
   * demo hace falta poder caer al simulador sin recargar la pagina si el
   * emparejamiento falla. */
  const usarDispositivoReal = React.useCallback((v: boolean) => {
    void transport.disconnect();
    setModoReal(v);
  }, [transport]);

  const connect = React.useCallback(() => {
    setHistory([]);
    setStats(emptyStats());
    streakRef.current = 0;
    void transport.connect();
  }, [transport]);

  const disconnect = React.useCallback(() => {
    void transport.disconnect();
  }, [transport]);

  const forceVerdict = React.useCallback(
    (v: VerdictKey) => {
      if ("setVerdict" in transport) (transport as MockTransport).setVerdict(v);
    },
    [transport],
  );

  const value = React.useMemo<TelemetryValue>(
    () => ({
      status,
      reading,
      frame,
      history,
      stats,
      inhaling,
      connect,
      disconnect,
      forceVerdict,
      modoReal,
      hayBluetooth,
      usarDispositivoReal,
    }),
    [status, reading, frame, history, stats, inhaling, connect, disconnect,
     forceVerdict, modoReal, hayBluetooth, usarDispositivoReal],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTelemetry(): TelemetryValue {
  const v = React.useContext(Ctx);
  if (!v) throw new Error("useTelemetry debe usarse dentro de TelemetryProvider");
  return v;
}
