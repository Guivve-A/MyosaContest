"use client";

import * as React from "react";
import { Award, BookOpen, Flame, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Bars, SparkColumns } from "@/components/viz/trace";
import { useTelemetry } from "@/hooks/use-telemetry";
import { cn } from "@/lib/utils";
import {
  VERDICTS,
  VERDICT_ORDER,
  formatClock,
  formatIe,
  verdictColorVar,
} from "@/lib/telemetry";

/** Sesiones previas. En producción vendrían del almacenamiento local o la nube. */
const HISTORICO = [42, 51, 48, 63, 58, 71, 66, 74, 69, 78, 82, 79, 86, 91];

const LECCIONES = [
  {
    titulo: "Respiración diafragmática",
    resumen:
      "Una mano en el abdomen, otra en el pecho. Al inhalar solo debe moverse la de abajo.",
    duracion: "3 min",
    completada: true,
  },
  {
    titulo: "Labios fruncidos",
    resumen:
      "Exhala por la boca entrecerrada durante el doble de tiempo que la inhalación.",
    duracion: "4 min",
    completada: true,
  },
  {
    titulo: "Por qué importa la relación I:E",
    resumen:
      "Una espiración prolongada reduce el atrapamiento aéreo y baja la frecuencia.",
    duracion: "5 min",
    completada: false,
  },
  {
    titulo: "Reconocer la respiración torácica",
    resumen:
      "Hombros que suben, esternón que bascula, sensación de falta de aire pese al esfuerzo.",
    duracion: "3 min",
    completada: false,
  },
];

/** Umbrales de gamificación, en porcentaje de tiempo en técnica correcta. */
const NIVELES = [
  { nombre: "Inicial", min: 0 },
  { nombre: "En progreso", min: 40 },
  { nombre: "Consistente", min: 65 },
  { nombre: "Dominio", min: 85 },
];

function nivelDe(pct: number) {
  return [...NIVELES].reverse().find((n) => pct >= n.min) ?? NIVELES[0];
}

export function AprendizajeTab() {
  const { stats, status } = useTelemetry();
  const connected = status.state === "connected";
  const pct = stats.onTechnique * 100;
  const nivel = nivelDe(pct);
  const siguiente = NIVELES.find((n) => n.min > pct);

  const reparto = VERDICT_ORDER.map((k) => ({
    label: VERDICTS[k].short,
    value: stats.perVerdict[k],
    color: verdictColorVar(VERDICTS[k].tone),
  }));

  const completadas = LECCIONES.filter((l) => l.completada).length;

  return (
    <div className="flex flex-col gap-4">
      {/* ---- Progreso y nivel ---- */}
      <Card className="animate-rise">
        <CardHeader className="flex-row items-start justify-between gap-3">
          <div className="flex flex-col gap-1">
            <CardTitle className="text-sm font-medium">
              Dominio de la técnica
            </CardTitle>
            <p className="text-xs text-muted-foreground">
              Basado en el tiempo evaluable de la sesión actual
            </p>
          </div>
          <Badge className="shrink-0">
            <Award className="size-3" data-icon="inline-start" />
            {nivel.nombre}
          </Badge>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <div className="flex items-end gap-3">
            <span className="tnum text-5xl font-semibold leading-none tracking-tight">
              {connected ? pct.toFixed(0) : "—"}
            </span>
            <span className="pb-1.5 text-sm text-muted-foreground">
              % en técnica correcta
            </span>
          </div>

          <div className="flex flex-col gap-1.5">
            <Progress value={pct} className="h-2" aria-label="Dominio" />
            <div className="flex justify-between text-[0.68rem] text-muted-foreground">
              <span>{nivel.nombre}</span>
              {siguiente ? (
                <span className="tnum">
                  {(siguiente.min - pct).toFixed(0)} puntos para {siguiente.nombre}
                </span>
              ) : (
                <span>Nivel máximo alcanzado</span>
              )}
            </div>
          </div>

          <Separator />

          <div className="grid grid-cols-3 gap-3">
            {[
              {
                icon: Flame,
                label: "Mejor racha",
                value: connected ? formatClock(stats.bestStreakS) : "—",
              },
              {
                icon: Target,
                label: "Respiraciones",
                value: connected ? String(stats.breaths) : "—",
              },
              {
                icon: BookOpen,
                label: "Lecciones",
                value: `${completadas}/${LECCIONES.length}`,
              },
            ].map((m, i) => (
              <div
                key={m.label}
                className={cn(
                  "animate-rise flex flex-col gap-1 rounded-lg bg-muted/50 p-3",
                  `stagger-${i + 1}`,
                )}
              >
                <m.icon className="size-3.5 text-muted-foreground" />
                <span className="tnum text-lg font-semibold leading-none">
                  {m.value}
                </span>
                <span className="text-[0.64rem] uppercase tracking-[0.1em] text-muted-foreground">
                  {m.label}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* ---- Resumen de sesión ---- */}
      <Card className="animate-rise stagger-2">
        <CardHeader>
          <CardTitle className="text-sm font-medium">
            Resumen de la sesión
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {[
              { l: "Duración", v: connected ? formatClock(stats.elapsedS) : "—" },
              {
                l: "Frecuencia media",
                v: connected && stats.meanBpm ? stats.meanBpm.toFixed(1) : "—",
                u: "rpm",
              },
              {
                l: "I:E media",
                v: connected && stats.meanIe ? formatIe(stats.meanIe) : "—",
              },
              {
                l: "Descartadas",
                v: connected ? String(Math.round(stats.discarded)) : "—",
                u: "vent.",
              },
            ].map((m) => (
              <div key={m.l} className="flex flex-col gap-0.5">
                <span className="text-[0.64rem] uppercase tracking-[0.11em] text-muted-foreground">
                  {m.l}
                </span>
                <span className="tnum text-base font-medium">
                  {m.v}
                  {m.u && (
                    <span className="ml-1 text-xs font-normal text-muted-foreground">
                      {m.u}
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>

          <Separator />

          <div className="flex flex-col gap-3">
            <span className="text-xs font-medium">Reparto por técnica</span>
            {stats.elapsedS > 0 ? (
              <Bars values={reparto} />
            ) : (
              <p className="py-2 text-sm text-muted-foreground">
                Sin datos todavía en esta sesión.
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* ---- Tendencia histórica ---- */}
      <Card className="animate-rise stagger-3">
        <CardHeader className="gap-1">
          <CardTitle className="text-sm font-medium">
            Tendencia · últimas {HISTORICO.length} sesiones
          </CardTitle>
          <p className="text-xs text-muted-foreground">
            Porcentaje de tiempo en técnica correcta por sesión
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <SparkColumns data={HISTORICO} color="var(--chart-1)" height={64} />
          <div className="flex items-center justify-between text-[0.68rem] text-muted-foreground">
            <span>Hace 14 sesiones</span>
            <span className="tnum font-medium text-foreground">
              +{HISTORICO[HISTORICO.length - 1] - HISTORICO[0]} puntos
            </span>
            <span>Hoy</span>
          </div>
        </CardContent>
      </Card>

      {/* ---- Lecciones ---- */}
      <Card className="animate-rise stagger-4">
        <CardHeader>
          <CardTitle className="text-sm font-medium">
            Aprende la técnica
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-0">
          {LECCIONES.map((l, i) => (
            <React.Fragment key={l.titulo}>
              {i > 0 && <Separator className="my-3" />}
              <div className="flex items-start gap-3">
                <div
                  className={cn(
                    "mt-0.5 grid size-7 shrink-0 place-items-center rounded-full text-[0.68rem] font-semibold",
                    l.completada
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground",
                  )}
                  aria-hidden
                >
                  {l.completada ? "✓" : i + 1}
                </div>
                <div className="flex flex-1 flex-col gap-0.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-sm font-medium">{l.titulo}</span>
                    <span className="shrink-0 text-[0.68rem] text-muted-foreground">
                      {l.duracion}
                    </span>
                  </div>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {l.resumen}
                  </p>
                </div>
              </div>
            </React.Fragment>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
