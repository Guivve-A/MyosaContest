"use client";

import * as React from "react";
import { Activity, MoveVertical, Timer, TrendingUp } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { BreathOrb } from "@/components/resumen/breath-orb";
import { useTelemetry } from "@/hooks/use-telemetry";
import { cn } from "@/lib/utils";
import {
  CONFIDENCE_FLOOR,
  VERDICTS,
  formatClock,
  formatIe,
  verdictColorVar,
} from "@/lib/telemetry";

function MetricTile({
  icon: Icon,
  label,
  value,
  unit,
  hint,
  className,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  className?: string;
}) {
  return (
    <Card className={cn("animate-rise gap-0 py-4", className)}>
      <CardContent className="flex flex-col gap-1.5 px-4">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Icon className="size-3.5" />
          <span className="text-[0.68rem] font-medium uppercase tracking-[0.11em]">
            {label}
          </span>
        </div>
        <div className="flex items-baseline gap-1">
          <span className="tnum text-2xl font-semibold leading-none">{value}</span>
          {unit && (
            <span className="text-xs text-muted-foreground">{unit}</span>
          )}
        </div>
        {hint && (
          <span className="text-[0.7rem] leading-snug text-muted-foreground">
            {hint}
          </span>
        )}
      </CardContent>
    </Card>
  );
}

export function ResumenTab() {
  const { status, reading, history, stats, inhaling } = useTelemetry();
  const connected = status.state === "connected";
  const meta = reading ? VERDICTS[reading.verdict] : null;

  // Por debajo del piso de confianza el firmware no coachea, y la app tampoco
  // debe: mostrar un veredicto dudoso como si fuera firme es peor que callar.
  const confident = !!reading && reading.confidence >= CONFIDENCE_FLOOR;

  return (
    <div className="flex flex-col gap-4">
      {/* ---- Veredicto principal ---- */}
      <Card className="animate-rise overflow-hidden py-0">
        <div className="grid gap-0 sm:grid-cols-[auto_1fr]">
          <div className="flex items-center justify-center bg-muted/40 p-5 sm:p-6">
            <BreathOrb
              verdict={confident ? reading!.verdict : null}
              inhaling={inhaling}
              bpm={reading?.bpm ?? 0}
              active={connected}
            />
          </div>

          <div className="flex flex-col justify-center gap-3 p-5 sm:p-6">
            <div className="flex flex-col gap-1.5">
              <span className="text-[0.68rem] font-medium uppercase tracking-[0.14em] text-muted-foreground">
                Técnica detectada
              </span>
              <div className="flex flex-wrap items-center gap-2.5">
                <h2
                  className="text-2xl font-semibold leading-tight tracking-tight sm:text-3xl"
                  style={{
                    color: meta && confident ? verdictColorVar(meta.tone) : undefined,
                  }}
                >
                  {!connected
                    ? "Sin conexión"
                    : confident
                      ? meta!.label
                      : "Evaluando"}
                </h2>
                {connected && confident && meta && (
                  <Badge
                    variant={meta.coachedOk ? "default" : "secondary"}
                    className="shrink-0"
                  >
                    {meta.coachedOk ? "Correcta" : "A corregir"}
                  </Badge>
                )}
              </div>
              <p className="max-w-prose text-sm leading-relaxed text-muted-foreground">
                {!connected
                  ? "Conecta el dispositivo para empezar a recibir biofeedback."
                  : confident
                    ? meta!.meaning
                    : "La confianza del modelo está por debajo del umbral. El dispositivo prefiere no evaluar antes que arriesgar una corrección equivocada."}
              </p>
            </div>

            {connected && reading && (
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between text-[0.7rem] text-muted-foreground">
                  <span>Confianza del modelo</span>
                  <span className="tnum">
                    {(reading.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <Progress
                  value={reading.confidence * 100}
                  className="h-1.5"
                  aria-label="Confianza del modelo"
                />
                <span className="text-[0.65rem] text-muted-foreground">
                  Umbral de coaching: {(CONFIDENCE_FLOOR * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* ---- Métricas ---- */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricTile
          icon={Activity}
          label="Frecuencia"
          value={connected && reading ? reading.bpm.toFixed(1) : "—"}
          unit="resp/min"
          hint="Objetivo en rehabilitación: 6 a 12"
          className="stagger-1"
        />
        <MetricTile
          icon={Timer}
          label="Relación I:E"
          value={connected && reading ? formatIe(reading.ieRatio) : "—"}
          hint="Exhalación larga: 1:2 o mayor"
          className="stagger-2"
        />
        <MetricTile
          icon={MoveVertical}
          label="Excursión torácica"
          value={connected && reading ? reading.tiltDeg.toFixed(2) : "—"}
          unit="grados"
          hint="Rotación del esternón"
          className="stagger-3"
        />
        <MetricTile
          icon={TrendingUp}
          label="En técnica"
          value={connected ? `${(stats.onTechnique * 100).toFixed(0)}` : "—"}
          unit="%"
          hint={`Sesión de ${formatClock(stats.elapsedS)}`}
          className="stagger-4"
        />
      </div>

      {/* ---- Aviso de señal alterada ---- */}
      {connected && reading?.verdict === "artifact" && (
        <Alert className="animate-rise">
          <AlertTitle>Señal alterada</AlertTitle>
          <AlertDescription>
            Se detectó tos, habla o movimiento. La evaluación queda suspendida
            hasta que la señal vuelva a ser limpia — no se registra como técnica
            incorrecta.
          </AlertDescription>
        </Alert>
      )}

      {/* ---- Línea de tiempo ---- */}
      <Card className="animate-rise stagger-5">
        <CardHeader>
          <CardTitle className="text-sm font-medium">
            Últimos {Math.min(history.length, 40) * 3} segundos
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {history.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">
              La sesión aún no registra veredictos.
            </p>
          ) : (
            <>
              <div className="flex h-12 items-end gap-[3px] overflow-hidden">
                {history.slice(-40).map((r, i) => (
                  <div
                    key={`${r.t}-${i}`}
                    title={`${VERDICTS[r.verdict].label} · ${r.bpm.toFixed(0)} rpm`}
                    className="animate-rise flex-1 rounded-[2px] transition-opacity hover:opacity-60"
                    style={{
                      height: `${28 + r.confidence * 72}%`,
                      background: verdictColorVar(VERDICTS[r.verdict].tone),
                      opacity: r.confidence >= CONFIDENCE_FLOOR ? 1 : 0.35,
                    }}
                  />
                ))}
              </div>
              <Separator />
              <div className="flex flex-wrap gap-x-4 gap-y-1.5">
                {Object.values(VERDICTS).map((v) => (
                  <span
                    key={v.key}
                    className="flex items-center gap-1.5 text-[0.68rem] text-muted-foreground"
                  >
                    <span
                      className="size-2 rounded-[2px]"
                      style={{ background: verdictColorVar(v.tone) }}
                    />
                    {v.short}
                  </span>
                ))}
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
