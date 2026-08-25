"use client";

/**
 * Orbe respiratorio.
 *
 * Es lo único de la interfaz que el paciente mira mientras respira, así que
 * carga dos trabajos a la vez: marcar el ritmo a seguir y decir si lo está
 * haciendo bien. El ritmo va en la escala (expande al inhalar, contrae al
 * exhalar) y el veredicto en el color.
 *
 * La transición de escala dura lo mismo que la fase respiratoria real, no un
 * valor fijo: si el paciente respira a 8 por minuto, el orbe tarda ~2.5 s en
 * expandirse. Un ritmo de animación que no coincide con el del paciente es peor
 * que ninguno, porque compite con su propia respiración en vez de acompañarla.
 */

import * as React from "react";

import { cn } from "@/lib/utils";
import { VERDICTS, verdictColorVar, type VerdictKey } from "@/lib/telemetry";

interface BreathOrbProps {
  verdict: VerdictKey | null;
  inhaling: boolean;
  bpm: number;
  active: boolean;
  className?: string;
}

export function BreathOrb({
  verdict,
  inhaling,
  bpm,
  active,
  className,
}: BreathOrbProps) {
  const meta = verdict ? VERDICTS[verdict] : null;
  const color = meta ? verdictColorVar(meta.tone) : "var(--muted-foreground)";

  // Duración de media respiración, acotada para que no quede ni frenética ni
  // catatónica si el BPM se dispara.
  const halfBreath = Math.min(6, Math.max(1.2, 30 / Math.max(4, bpm)));

  return (
    <div
      className={cn(
        "relative grid aspect-square w-full max-w-[248px] place-items-center",
        className,
      )}
    >
      {/* Halo exterior: presencia ambiental, no marca el ritmo. */}
      <div
        aria-hidden
        className={cn(
          "absolute inset-2 rounded-full blur-2xl",
          active && "animate-halo",
        )}
        style={{ background: color, opacity: active ? undefined : 0.08 }}
      />

      {/* Anillos concéntricos estáticos: dan escala y profundidad sin moverse. */}
      <svg viewBox="0 0 200 200" className="absolute inset-0 size-full" aria-hidden>
        <circle cx="100" cy="100" r="94" fill="none" stroke="var(--border)" strokeWidth="1" />
        <circle
          cx="100"
          cy="100"
          r="74"
          fill="none"
          stroke="var(--border)"
          strokeWidth="1"
          strokeDasharray="2 6"
        />
      </svg>

      {/* El orbe. La escala sigue la fase respiratoria real. */}
      <div
        data-breath-orb
        className="relative grid size-[62%] place-items-center rounded-full transition-transform ease-in-out"
        style={{
          background: `radial-gradient(circle at 32% 28%, ${color}, color-mix(in oklch, ${color} 62%, var(--background)))`,
          transform: active ? `scale(${inhaling ? 1.05 : 0.84})` : "scale(0.9)",
          transitionDuration: `${halfBreath}s`,
          boxShadow: active
            ? `0 0 0 1px color-mix(in oklch, ${color} 30%, transparent), 0 18px 48px -12px ${color}`
            : "none",
          opacity: active ? 1 : 0.35,
        }}
      >
        <div className="flex flex-col items-center gap-0.5 text-center">
          <span
            className="text-[0.62rem] font-medium uppercase tracking-[0.16em]"
            style={{ color: "var(--primary-foreground)" }}
          >
            {active ? (inhaling ? "Inhala" : "Exhala") : "En pausa"}
          </span>
          {active && (
            <span
              className="tnum text-3xl font-semibold leading-none"
              style={{ color: "var(--primary-foreground)" }}
            >
              {bpm.toFixed(0)}
            </span>
          )}
          {active && (
            <span
              className="text-[0.6rem] uppercase tracking-[0.12em] opacity-75"
              style={{ color: "var(--primary-foreground)" }}
            >
              resp/min
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
