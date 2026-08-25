"use client";

/**
 * Traza de señal en SVG.
 *
 * Sin librería de gráficos a propósito: son polilíneas de una sola serie que se
 * repintan a 30 fps. Recharts o similares montarían un árbol de componentes por
 * cuadro para dibujar lo que aquí es una cadena de puntos, y el coste se nota
 * justo en el móvil de gama media que va a llevar el paciente.
 */

import * as React from "react";

import { cn } from "@/lib/utils";

interface TraceProps {
  data: number[];
  /** Escala vertical. Si se omite, se autoajusta al máximo absoluto visible. */
  range?: number;
  color?: string;
  className?: string;
  /** Rellena bajo la curva con un degradado hasta transparente. */
  fill?: boolean;
  strokeWidth?: number;
  /** Marca el último punto con un círculo pulsante. */
  showHead?: boolean;
  height?: number;
  "aria-label"?: string;
}

export function Trace({
  data,
  range,
  color = "var(--chart-1)",
  className,
  fill = true,
  strokeWidth = 1.75,
  showHead = true,
  height = 72,
  "aria-label": ariaLabel,
}: TraceProps) {
  const W = 300;
  const H = height;
  const id = React.useId();

  const { d, area, headY } = React.useMemo(() => {
    if (data.length < 2) return { d: "", area: "", headY: H / 2 };

    const amp = range ?? Math.max(0.001, ...data.map(Math.abs)) * 1.15;
    const step = W / (data.length - 1);
    const y = (v: number) => H / 2 - (v / amp) * (H / 2 - 3);

    let path = `M0,${y(data[0]).toFixed(2)}`;
    for (let i = 1; i < data.length; i++) {
      path += ` L${(i * step).toFixed(2)},${y(data[i]).toFixed(2)}`;
    }
    return {
      d: path,
      area: `${path} L${W},${H} L0,${H} Z`,
      headY: y(data[data.length - 1]),
    };
  }, [data, range, H]);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className={cn("w-full", className)}
      style={{ height }}
      role="img"
      aria-label={ariaLabel}
    >
      <defs>
        <linearGradient id={`g-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.22" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>

      <line
        x1="0"
        y1={H / 2}
        x2={W}
        y2={H / 2}
        stroke="var(--signal-grid)"
        strokeWidth="1"
        strokeDasharray="3 5"
        vectorEffect="non-scaling-stroke"
      />

      {fill && d && <path d={area} fill={`url(#g-${id})`} />}
      {d && (
        <path
          d={d}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      )}
      {showHead && d && (
        <circle cx={W - 1} cy={headY} r="3" fill={color} vectorEffect="non-scaling-stroke" />
      )}
    </svg>
  );
}

/* -------------------------------------------------------------------------- */

interface BarsProps {
  values: { label: string; value: number; color?: string }[];
  max?: number;
  className?: string;
}

/** Barras horizontales para distribuciones pequeñas (reparto por técnica). */
export function Bars({ values, max, className }: BarsProps) {
  const top = max ?? Math.max(1, ...values.map((v) => v.value));
  return (
    <div className={cn("flex flex-col gap-2.5", className)}>
      {values.map((v, i) => (
        <div key={v.label} className="flex items-center gap-3">
          <span className="w-24 shrink-0 truncate text-xs text-muted-foreground">
            {v.label}
          </span>
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
            <div
              className={cn("h-full rounded-full transition-[width] duration-700 ease-out", `stagger-${i + 1}`)}
              style={{
                width: `${Math.max(2, (v.value / top) * 100)}%`,
                background: v.color ?? "var(--chart-1)",
              }}
            />
          </div>
          <span className="tnum w-10 shrink-0 text-right text-xs text-muted-foreground">
            {Math.round((v.value / top) * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------------------- */

interface SparkColumnsProps {
  data: number[];
  color?: string;
  className?: string;
  height?: number;
}

/** Columnas compactas para tendencias históricas por sesión. */
export function SparkColumns({
  data,
  color = "var(--chart-1)",
  className,
  height = 56,
}: SparkColumnsProps) {
  const max = Math.max(1, ...data);
  return (
    <div
      className={cn("flex items-end gap-1", className)}
      style={{ height }}
      role="img"
      aria-label={`Tendencia de ${data.length} sesiones`}
    >
      {data.map((v, i) => (
        <div
          key={i}
          className="animate-rise flex-1 rounded-sm transition-opacity hover:opacity-70"
          style={{
            height: `${Math.max(6, (v / max) * 100)}%`,
            background: color,
            opacity: 0.35 + (v / max) * 0.65,
            animationDelay: `${i * 22}ms`,
          }}
        />
      ))}
    </div>
  );
}
