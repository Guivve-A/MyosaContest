"use client";

import * as React from "react";
import { Battery, Bluetooth, BluetoothConnected, Signal } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useTelemetry } from "@/hooks/use-telemetry";
import { cn } from "@/lib/utils";
import { VERDICT_ORDER, VERDICTS, verdictColorVar } from "@/lib/telemetry";

const ESTADO_TEXTO = {
  disconnected: "Desconectado",
  scanning: "Buscando…",
  connecting: "Conectando…",
  connected: "Conectado",
} as const;

export function DeviceBar() {
  const { status, connect, disconnect, forceVerdict,
          modoReal, hayBluetooth, usarDispositivoReal } = useTelemetry();
  const [error, setError] = React.useState<string | null>(null);
  const connected = status.state === "connected";
  const busy = status.state === "scanning" || status.state === "connecting";

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        {/* Indicador de enlace */}
        <div className="relative grid size-9 shrink-0 place-items-center rounded-full border bg-card">
          {connected ? (
            <BluetoothConnected className="size-4 text-primary" />
          ) : (
            <Bluetooth
              className={cn(
                "size-4 text-muted-foreground",
                busy && "animate-pulse",
              )}
            />
          )}
          {connected && (
            <span
              aria-hidden
              className="animate-ping-soft absolute inset-0 rounded-full ring-1 ring-primary"
            />
          )}
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
          <span className="truncate text-sm font-medium leading-tight">
            {status.name}
          </span>
          <span className="text-[0.7rem] leading-tight text-muted-foreground">
            {ESTADO_TEXTO[status.state]}
            {connected && ` · ${status.effectiveHz.toFixed(2)} Hz`}
            {/* Que los datos sean simulados NO se insinua: se dice. Una demo
                que no distingue una medicion de una simulacion es exactamente
                lo que este proyecto lleva evitando. */}
            {!modoReal && (
              <span className="ml-1.5 rounded bg-muted px-1.5 py-0.5 text-[10px]
                               font-medium uppercase tracking-wide">
                simulado
              </span>
            )}
          </span>
        </div>

        {connected && (
          <div className="hidden items-center gap-3 sm:flex">
            <Tooltip>
              <TooltipTrigger
                render={
                  <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Signal className="size-3.5" />
                    <span className="tnum">{status.rssi}</span>
                  </span>
                }
              />
              <TooltipContent>Intensidad de señal (dBm)</TooltipContent>
            </Tooltip>
            <Separator orientation="vertical" className="h-4" />
            <Tooltip>
              <TooltipTrigger
                render={
                  <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Battery className="size-3.5" />
                    <span className="tnum">{status.battery}%</span>
                  </span>
                }
              />
              <TooltipContent>Batería del dispositivo</TooltipContent>
            </Tooltip>
          </div>
        )}

        {/* Alternar dispositivo real / simulador. En Safari o Firefox no hay
            Web Bluetooth, asi que el boton se explica en vez de fallar. */}
        {!connected && (
          <Button
            size="sm"
            variant="ghost"
            className="text-xs"
            disabled={!hayBluetooth && !modoReal}
            onClick={() => { setError(null); usarDispositivoReal(!modoReal); }}
          >
            {modoReal ? "Usar simulador" : "Usar dispositivo"}
          </Button>
        )}
        <Button
          size="sm"
          variant={connected ? "outline" : "default"}
          onClick={async () => {
            setError(null);
            if (connected) { disconnect(); return; }
            try { await Promise.resolve(connect()); }
            catch (e) { setError(e instanceof Error ? e.message : String(e)); }
          }}
          disabled={busy}
          className="shrink-0"
        >
          {connected ? "Desconectar" : busy ? "…" : "Conectar"}
        </Button>
      </div>

      {/* Control de demostración. En producción esto no existe: la técnica la
          determina el modelo en el dispositivo, no la interfaz. Se mantiene
          visible y etiquetado para poder ensayar la demo sin un paciente. */}
      {connected && (
        <div className="flex flex-wrap items-center gap-1.5 rounded-lg border border-dashed bg-muted/30 p-2">
          <span className="px-1 text-[0.62rem] uppercase tracking-[0.12em] text-muted-foreground">
            Demo
          </span>
          {VERDICT_ORDER.map((k) => (
            <Button
              key={k}
              size="sm"
              variant="ghost"
              className="h-7 px-2 text-[0.7rem]"
              onClick={() => forceVerdict(k)}
            >
              <span
                className="size-1.5 rounded-full"
                data-icon="inline-start"
                style={{ background: verdictColorVar(VERDICTS[k].tone) }}
              />
              {VERDICTS[k].short}
            </Button>
          ))}
        </div>
      )}
    </div>
  );
}

export function SessionBadge() {
  const { status } = useTelemetry();
  if (status.state !== "connected") return null;
  return (
    <Badge variant="secondary" className="font-normal">
      Sesión activa
    </Badge>
  );
}
