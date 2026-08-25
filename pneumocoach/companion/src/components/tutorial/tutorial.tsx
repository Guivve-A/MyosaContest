"use client";

/**
 * Tutorial de tres pasos: conectar, colocar y calibrar, y la sesión.
 *
 * Las animaciones son SVG animado con `motion`, no ficheros Lottie. La razón es
 * concreta: un Lottie es un JSON exportado de After Effects que hay que
 * mantener fuera del repositorio y que no puede leer el estado de la app. Estas
 * animaciones sí — la del pecho respira al ritmo real que el dispositivo está
 * midiendo, y la de conexión refleja si hay dispositivo o no.
 *
 * `lottie-react` está instalado por si más adelante el equipo de diseño entrega
 * animaciones autoría suya; se dejarían caer aquí sin tocar la estructura.
 *
 * Qué enseña y por qué en este orden
 * ----------------------------------
 * Los tres pasos son los tres sitios donde una demo se rompe: el paciente no
 * encuentra el dispositivo, se lo coloca mal, o no sabe que la sesión ya empezó
 * y se mueve. El tutorial existe para cada uno de esos fallos, no para adornar.
 */

import * as React from "react";
import { motion } from "motion/react";

/* Nota sobre el cambio de paso.
 *
 * La primera version usaba <AnimatePresence mode="wait">. Con React 19 se
 * quedaba atascada: el contador avanzaba a "Paso 3 de 3" y el contenido seguia
 * mostrando el paso 1, porque la animacion de salida no completaba nunca y el
 * hijo viejo no llegaba a desmontarse.
 *
 * Un `key` sobre el motion.div hace que React remonte el bloque y que la
 * animacion de entrada se reproduzca sola. Se pierde la transicion de salida,
 * que aqui no aporta nada, y se gana que no exista un estado en el que la
 * interfaz muestre un paso y diga otro -que en un tutorial es lo peor que
 * puede pasar-. */

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

/* -------------------------------------------------------------------------- */
/* Paso 1 · Conexión                                                           */
/* -------------------------------------------------------------------------- */

function AnimacionConexion({ conectado }: { conectado: boolean }) {
  return (
    <svg viewBox="0 0 320 160" className="w-full" role="img"
         aria-label="El teléfono se conecta al dispositivo por Bluetooth">
      {/* Dispositivo */}
      <motion.rect
        x="18" y="52" width="62" height="62" rx="14"
        className="fill-foreground/90"
        animate={{ y: [52, 49, 52] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
      />
      <rect x="30" y="64" width="38" height="26" rx="5" className="fill-background" />
      <circle cx="49" cy="103" r="3.5" className="fill-primary" />

      {/* Ondas: nacen en el dispositivo y viajan al teléfono */}
      {[0, 1, 2].map((i) => (
        <motion.circle
          key={i}
          cx="86" cy="83" r="10"
          className="fill-none stroke-primary"
          strokeWidth={2}
          initial={{ scale: 0.4, opacity: 0 }}
          animate={
            conectado
              ? { scale: [0.4, 2.6], opacity: [0, 0.55, 0] }
              : { scale: 0.4, opacity: 0.12 }
          }
          transition={{
            duration: 2.1, repeat: Infinity, delay: i * 0.7, ease: "easeOut",
          }}
          style={{ originX: "86px", originY: "83px" }}
        />
      ))}

      {/* Teléfono */}
      <rect x="232" y="34" width="70" height="98" rx="12"
            className="fill-none stroke-foreground/40" strokeWidth={2.5} />
      <rect x="240" y="46" width="54" height="66" rx="4"
            className="fill-foreground/5" />
      <motion.rect
        x="240" y="46" width="54" height="66" rx="4"
        className="fill-primary"
        animate={{ opacity: conectado ? [0.08, 0.22, 0.08] : 0 }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
      />
      <rect x="256" y="121" width="22" height="3" rx="1.5"
            className="fill-foreground/30" />
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/* Paso 2 · Colocación                                                         */
/* -------------------------------------------------------------------------- */

/**
 * La posición del sensor no es decorativa: el centro va sobre el manubrio, unos
 * 35 mm bajo la escotadura yugular. Más abajo, sobre el apéndice xifoides, la
 * relación entre los dos canales se invierte y el modelo lee al revés. La
 * animación marca ese punto explícitamente por eso.
 */
function AnimacionColocacion() {
  return (
    <svg viewBox="0 0 320 160" className="w-full" role="img"
         aria-label="El sensor se coloca sobre el esternón, bajo la escotadura yugular">
      {/* Torso */}
      <path
        d="M160 18c-14 0-22 6-24 14l-30 12c-8 3-12 9-12 18v82h132V62c0-9-4-15-12-18l-30-12c-2-8-10-14-24-14z"
        className="fill-foreground/8 stroke-foreground/25" strokeWidth={2}
      />
      {/* Escotadura yugular, la referencia anatómica */}
      <circle cx="160" cy="40" r="3" className="fill-foreground/40" />
      <text x="168" y="38" className="fill-muted-foreground" fontSize="9">
        escotadura
      </text>

      {/* Guía de los 35 mm */}
      <motion.line
        x1="160" y1="43" x2="160" y2="70"
        className="stroke-primary" strokeWidth={1.4} strokeDasharray="3 3"
        animate={{ opacity: [0.3, 0.9, 0.3] }}
        transition={{ duration: 2.4, repeat: Infinity }}
      />
      <text x="167" y="60" className="fill-primary" fontSize="9">35 mm</text>

      {/* El sensor, que respira con el torso */}
      <motion.g
        animate={{ y: [0, -2.4, 0], rotate: [0, -1.6, 0] }}
        transition={{ duration: 4.2, repeat: Infinity, ease: "easeInOut" }}
        style={{ originX: "160px", originY: "82px" }}
      >
        <rect x="143" y="66" width="34" height="34" rx="9"
              className="fill-foreground/90" />
        <rect x="143" y="66" width="34" height="34" rx="9"
              className="fill-none stroke-primary" strokeWidth={1.6} />
      </motion.g>

      {/* Las dos manos de la consigna */}
      <motion.ellipse
        cx="112" cy="104" rx="15" ry="10"
        className="fill-foreground/15 stroke-foreground/35" strokeWidth={1.5}
        animate={{ cx: [112, 112, 112] }}
      />
      <text x="96" y="124" className="fill-muted-foreground" fontSize="8.5">
        pecho: quieta
      </text>
      <motion.ellipse
        cx="208" cy="120" rx="15" ry="10"
        className="fill-primary/25 stroke-primary" strokeWidth={1.5}
        animate={{ cy: [120, 113, 120], ry: [10, 11.5, 10] }}
        transition={{ duration: 4.2, repeat: Infinity, ease: "easeInOut" }}
      />
      <text x="188" y="142" className="fill-primary" fontSize="8.5">
        abdomen: se mueve
      </text>
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/* Paso 3 · La sesión                                                          */
/* -------------------------------------------------------------------------- */

const FASES = [
  { nombre: "Asentando", seg: 20, nota: "colócate y quédate quieto" },
  { nombre: "Prepárate", seg: 10, nota: "una mano al pecho, otra al abdomen" },
  { nombre: "Abdomen", seg: 30, nota: "respira moviendo solo el abdomen" },
  { nombre: "Pecho", seg: 30, nota: "ahora solo el pecho" },
  { nombre: "Calibrado", seg: 8, nota: "tu eje personal, ya medido" },
  { nombre: "Sesión", seg: 0, nota: "el dispositivo te lee en vivo" },
];

function AnimacionSesion() {
  const [activa, setActiva] = React.useState(0);
  React.useEffect(() => {
    const t = setInterval(() => setActiva((a) => (a + 1) % FASES.length), 1800);
    return () => clearInterval(t);
  }, []);
  const total = FASES.reduce((s, f) => s + f.seg, 0);

  return (
    <div className="w-full space-y-3">
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-muted">
        {FASES.filter((f) => f.seg > 0).map((f, i) => (
          <motion.div
            key={f.nombre}
            className={i === activa ? "bg-primary" : "bg-primary/25"}
            style={{ width: `${(f.seg / total) * 100}%` }}
            animate={{ opacity: i === activa ? 1 : 0.55 }}
            transition={{ duration: 0.4 }}
          />
        ))}
      </div>
      <div className="min-h-14">
        <motion.div
          key={activa}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
            <p className="text-base font-semibold">
              {FASES[activa].nombre}
              {FASES[activa].seg > 0 && (
                <span className="ml-2 text-sm font-normal text-muted-foreground">
                  {FASES[activa].seg} s
                </span>
              )}
            </p>
            <p className="text-sm text-muted-foreground">{FASES[activa].nota}</p>
        </motion.div>
      </div>
      <p className="text-xs text-muted-foreground">
        La sesión avanza sola. No tienes que pulsar nada: estás respirando con
        las dos manos ocupadas, y cualquier cosa que acciones se mide como
        movimiento.
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

const PASOS = [
  {
    titulo: "Conecta el dispositivo",
    cuerpo:
      "Enciende el PneumoCoach y pulsa Conectar. Aparecerá en la lista como " +
      "«PneumoCoach». Hace falta Chrome en Android: Safari no tiene Bluetooth web.",
    render: (p: { conectado: boolean }) => <AnimacionConexion {...p} />,
  },
  {
    titulo: "Colócate el sensor",
    cuerpo:
      "Sobre el esternón, en la línea media, unos 35 mm por debajo de la " +
      "escotadura yugular. Más abajo la señal se invierte y el sistema lee al revés.",
    render: () => <AnimacionColocacion />,
  },
  {
    titulo: "Deja que se calibre",
    cuerpo:
      "El aparato mide TU eje: primero respiras solo con el abdomen, después " +
      "solo con el pecho. A partir de ahí sabe distinguir una de otra en ti.",
    render: () => <AnimacionSesion />,
  },
];

export function Tutorial({
  conectado,
  onCerrar,
}: {
  conectado: boolean;
  onCerrar: () => void;
}) {
  const [paso, setPaso] = React.useState(0);
  const ultimo = paso === PASOS.length - 1;

  return (
    <Card className="mx-auto w-full max-w-md overflow-hidden p-5">
      <div className="mb-1 flex items-center gap-1.5">
        {PASOS.map((_, i) => (
          <motion.div
            key={i}
            className="h-1 flex-1 rounded-full bg-primary"
            animate={{ opacity: i <= paso ? 1 : 0.2 }}
            transition={{ duration: 0.3 }}
          />
        ))}
      </div>
      <p className="mb-3 text-xs uppercase tracking-wider text-muted-foreground">
        Paso {paso + 1} de {PASOS.length}
      </p>

      <div className="mb-4 flex min-h-40 items-center justify-center">
        <motion.div
          key={paso}
          className="w-full"
          initial={{ opacity: 0, x: 24 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.32, ease: "easeOut" }}
        >
          {PASOS[paso].render({ conectado })}
        </motion.div>
      </div>

      <motion.div
        key={`t${paso}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.25 }}
      >
        <h2 className="text-lg font-semibold">{PASOS[paso].titulo}</h2>
        <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
          {PASOS[paso].cuerpo}
        </p>
      </motion.div>

      <div className="mt-5 flex gap-2">
        {paso > 0 && (
          <Button variant="outline" className="flex-1"
                  onClick={() => setPaso((p) => p - 1)}>
            Atrás
          </Button>
        )}
        <Button
          className="flex-1"
          onClick={() => (ultimo ? onCerrar() : setPaso((p) => p + 1))}
        >
          {ultimo ? "Empezar" : "Siguiente"}
        </Button>
      </div>
      {!ultimo && (
        <button
          onClick={onCerrar}
          className="mt-2 w-full text-xs text-muted-foreground underline-offset-2 hover:underline"
        >
          Saltar tutorial
        </button>
      )}
    </Card>
  );
}
