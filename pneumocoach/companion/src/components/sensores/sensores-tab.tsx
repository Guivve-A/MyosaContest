"use client";

import * as React from "react";
import { Gauge, RotateCcw, Thermometer, Waves } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Trace } from "@/components/viz/trace";
import { useTelemetry } from "@/hooks/use-telemetry";
import { cn } from "@/lib/utils";
import { FS_HZ, WINDOW_S } from "@/lib/telemetry";

function Readout({
  label,
  value,
  unit,
  mono = true,
}: {
  label: string;
  value: string;
  unit?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[0.64rem] uppercase tracking-[0.11em] text-muted-foreground">
        {label}
      </span>
      <span className={cn("text-sm font-medium", mono && "tnum")}>
        {value}
        {unit && (
          <span className="ml-1 text-xs font-normal text-muted-foreground">
            {unit}
          </span>
        )}
      </span>
    </div>
  );
}

export function SensoresTab() {
  const { status, frame, reading } = useTelemetry();
  const connected = status.state === "connected";

  const dropRate = status.effectiveHz
    ? Math.max(0, (1 - status.effectiveHz / FS_HZ) * 100)
    : 0;

  return (
    <div className="flex flex-col gap-4">
      {/* ---- Canales mecánicos ---- */}
      <Card className="animate-rise">
        <CardHeader className="gap-1">
          <CardTitle className="text-sm font-medium">
            Movimiento tóraco-abdominal
          </CardTitle>
          {/* Este texto decía que el cociente de los dos canales «es lo que
              separa una técnica correcta de una torácica». Eso se midió sobre
              un pecho real y es falso (ADR-0006): reproducirlo con
              `python tools/medir_cociente.py`. Quien clasifica es el modelo,
              no ninguna de las dos trazas por separado. */}
          <p className="text-xs leading-relaxed text-muted-foreground">
            La señal del IMU se descompone en dos movimientos mecánicamente
            distintos. Los dos alimentan al modelo y ninguno decide por sí
            solo: el veredicto sale de 29 características proyectadas sobre el
            eje de referencia del propio paciente — no de la frecuencia
            respiratoria, y no de estas trazas leídas a ojo.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <div className="flex items-baseline justify-between">
              <div className="flex items-center gap-2">
                <span
                  className="size-2 rounded-full"
                  style={{ background: "var(--chart-1)" }}
                />
                <span className="text-xs font-medium">Inclinación</span>
                <span className="text-[0.68rem] text-muted-foreground">
                  rotación del esternón
                </span>
              </div>
              <span className="tnum text-xs text-muted-foreground">
                {reading ? `${reading.tiltDeg.toFixed(2)}°` : "—"}
              </span>
            </div>
            <Trace
              data={frame?.tilt ?? []}
              color="var(--chart-1)"
              height={78}
              aria-label="Traza del canal de inclinación"
            />
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex items-baseline justify-between">
              <div className="flex items-center gap-2">
                <span
                  className="size-2 rounded-full"
                  style={{ background: "var(--chart-2)" }}
                />
                <span className="text-xs font-medium">Axial</span>
                <span className="text-[0.68rem] text-muted-foreground">
                  traslación antero-posterior
                </span>
              </div>
              <span className="tnum text-xs text-muted-foreground">
                {reading ? `${reading.axialMg.toFixed(2)} mg` : "—"}
              </span>
            </div>
            <Trace
              data={frame?.axial ?? []}
              color="var(--chart-2)"
              height={78}
              aria-label="Traza del canal axial"
            />
          </div>

          <Separator />

          <div className="flex flex-col gap-2">
            <div className="flex items-baseline justify-between">
              <span className="text-xs font-medium">
                Cociente inclinación / axial
              </span>
              <span className="tnum text-xs text-muted-foreground">
                {reading ? reading.ratio.toFixed(2) : "—"}
              </span>
            </div>
            {/* El sentido se dice con PALABRAS, no con una d con signo: el
                repositorio usa dos convenios opuestos -analizar_captura.py
                resta torácica menos diafragmática y diagnostico_premisa.py al
                revés-, así que una d suelta aquí no significaría nada. */}
            <p className="text-[0.7rem] leading-relaxed text-muted-foreground">
              Este cociente era el discriminador del diseño original. Medido
              sobre un pecho real no lo es: en las cuatro sesiones se inclina al
              revés de lo previsto — la respiración diafragmática marca más
              cociente que la torácica — y su magnitud varía entre sesiones en
              dos órdenes de magnitud. Los dos canales suben juntos y la
              división los cancela. Se muestra como dato del canal, no como
              veredicto (ADR-0006).
            </p>
          </div>
        </CardContent>
      </Card>

      {/* ---- Postura y ambiente ---- */}
      <div className="grid gap-3 md:grid-cols-2">
        <Card className="animate-rise stagger-1">
          <CardHeader className="flex-row items-center gap-2">
            <RotateCcw className="size-4 text-muted-foreground" />
            <CardTitle className="text-sm font-medium">Postura</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4">
              <Readout
                label="Cabeceo"
                value={connected && frame ? frame.pitch.toFixed(1) : "—"}
                unit="°"
              />
              <Readout
                label="Giroscopio"
                value={connected && frame ? frame.gyroRms.toFixed(2) : "—"}
                unit="dps"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-[0.68rem] text-muted-foreground">
                <span>Estabilidad postural</span>
                <span className="tnum">
                  {connected && frame
                    ? `${Math.max(0, 100 - frame.gyroRms * 9).toFixed(0)}%`
                    : "—"}
                </span>
              </div>
              <Progress
                value={connected && frame ? Math.max(0, 100 - frame.gyroRms * 9) : 0}
                className="h-1.5"
                aria-label="Estabilidad postural"
              />
            </div>
            <p className="text-[0.7rem] leading-relaxed text-muted-foreground">
              El cabeceo se estima fusionando acelerómetro y giroscopio con un
              filtro complementario. La estabilidad baja cuando el paciente se
              reacomoda, y ahí las ventanas se descartan.
            </p>
          </CardContent>
        </Card>

        <Card className="animate-rise stagger-2">
          <CardHeader className="flex-row items-center gap-2">
            <Gauge className="size-4 text-muted-foreground" />
            <CardTitle className="text-sm font-medium">Ambiente</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4">
              <Readout
                label="Presión"
                value={connected && frame ? frame.pressureHpa.toFixed(1) : "—"}
                unit="hPa"
              />
              <Readout
                label="Temperatura"
                value={connected && frame ? frame.tempC.toFixed(1) : "—"}
                unit="°C"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Readout
                label="Altitud rel."
                value={connected && frame ? frame.altitudeM.toFixed(1) : "—"}
                unit="m"
              />
              <div className="flex flex-col gap-0.5">
                <span className="text-[0.64rem] uppercase tracking-[0.11em] text-muted-foreground">
                  Sensor
                </span>
                <span className="text-sm font-medium">BMP180</span>
              </div>
            </div>
            <p className="text-[0.7rem] leading-relaxed text-muted-foreground">
              El barómetro aporta contexto, no señal respiratoria: detecta
              cambios de altitud y deriva térmica que podrían alterar la lectura
              del acelerómetro a lo largo de una sesión larga.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* ---- Calidad de adquisición ---- */}
      <Card className="animate-rise stagger-3">
        <CardHeader className="flex-row items-center gap-2">
          <Waves className="size-4 text-muted-foreground" />
          <CardTitle className="text-sm font-medium">
            Calidad de adquisición
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Readout
              label="Tasa efectiva"
              value={connected ? status.effectiveHz.toFixed(2) : "—"}
              unit="Hz"
            />
            <Readout
              label="Objetivo"
              value={FS_HZ.toFixed(0)}
              unit="Hz"
            />
            <Readout
              label="Ventana"
              value={WINDOW_S.toFixed(0)}
              unit="s"
            />
            <Readout
              label="Pérdidas"
              value={connected ? dropRate.toFixed(2) : "—"}
              unit="%"
            />
          </div>

          <Separator />

          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary" className="font-normal">
              <Thermometer className="size-3" data-icon="inline-start" />
              IMU {status.imu}
            </Badge>
            <Badge variant="secondary" className="font-normal">
              Anti-aliasing 21 Hz
            </Badge>
            <Badge variant="secondary" className="font-normal">
              Paso banda 0.1–1 Hz
            </Badge>
            <Badge variant="secondary" className="font-normal">
              Firmware {status.firmware}
            </Badge>
          </div>

          <p className="text-[0.7rem] leading-relaxed text-muted-foreground">
            El muestreo va anclado a un núcleo dedicado del ESP32 con
            planificación por instantes absolutos. El jitter temporal se
            traduciría en contaminación del espectro justo en la banda
            respiratoria, así que la tasa efectiva es una métrica de salud del
            sistema y no un dato decorativo.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
