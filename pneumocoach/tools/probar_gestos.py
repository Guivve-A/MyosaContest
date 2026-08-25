"""Prueba de gestos del APDS9960, guiada y sin depender de nadie más.

    python tools/probar_gestos.py --puerto COM9

Solo hace falta la placa conectada por USB. No hay que ponerse nada.

Qué comprueba, en dos fases
---------------------------
1. PROXIMIDAD. ¿El sensor ve la mano? Es la pregunta que hay que responder
   primero, porque «no detecta gestos» tiene dos causas que desde fuera se ven
   igual: que el sensor no vea nada, o que lo vea y el decodificador no saque
   dirección. Si la proximidad no sube al acercar la mano, no tiene sentido
   probar direcciones.

2. DIRECCIÓN. Pide los cuatro gestos de uno en uno y compara lo detectado con
   lo pedido. Lo que importa no es sólo que detecte, sino que **la dirección
   coincida**: los ejes del sensor pueden estar rotados respecto a cómo quedó
   montada la placa, y un «arriba» que sale como «izquierda» hay que corregirlo
   en el mapeo, no en la mano del paciente.

Al final imprime la matriz de lo pedido contra lo detectado. Si hay una rotación
constante, se ve de un vistazo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401
from paridad import abrir  # noqa: E402

DIRECCIONES = [
    ("ARRIBA", "de ABAJO hacia ARRIBA", "es el que avanza la sesión"),
    ("ABAJO", "de ARRIBA hacia ABAJO", "termina la sesión"),
    ("DERECHA", "de IZQUIERDA a DERECHA", "sin acción asignada"),
    ("IZQUIERDA", "de DERECHA a IZQUIERDA", "sin acción asignada"),
]

REPETICIONES = 3
ESPERA_GESTO = 12.0     # segundos por intento
UMBRAL_PROX = 40        # el mismo GPENTH que configura el firmware


class Lector:
    """Lee el puerto una sola vez y reparte las líneas por prefijo."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = b""
        self.gestos: list[str] = []
        self.prox: list[tuple[int, int, int, int]] = []
        self.otras: list[str] = []

    def drena(self) -> None:
        self.buf += self.ser.read(self.ser.in_waiting or 0)
        while b"\n" in self.buf:
            ln, self.buf = self.buf.split(b"\n", 1)
            t = ln.decode("utf-8", "replace").strip()
            if not t:
                continue
            if t.startswith("G,"):
                self.gestos.append(t[2:].strip())
            elif t.startswith("P,"):
                p = t.split(",")
                if len(p) == 5:
                    try:
                        self.prox.append(tuple(int(v) for v in p[1:]))
                    except ValueError:
                        pass
            elif t.startswith("#"):
                self.otras.append(t)


def barra(v: int, ancho: int = 34, tope: int = 255) -> str:
    n = max(0, min(ancho, round(v * ancho / tope)))
    return "[" + "#" * n + "·" * (ancho - n) + "]"


def fase_proximidad(ser, lector: Lector) -> bool:
    print("=" * 68)
    print("  FASE 1 · ¿El sensor ve tu mano?")
    print("=" * 68)
    print()
    print("  Acerca la mano al módulo APDS9960 y aléjala, varias veces,")
    print("  entre unos 3 y 20 cm. Tienes 25 segundos.")
    print()
    input("  Pulsa Enter cuando estés listo... ")
    print()

    lector.prox.clear()
    ser.write(b"p")
    # Los dos primeros segundos son la linea base: sirve de referencia para que
    # el maximo signifique algo en vez de ser un numero suelto.
    base_hasta = time.time() + 2.0
    base = []
    t0 = time.time()
    pmax = 0
    while time.time() - t0 < 27:
        lector.drena()
        if lector.prox:
            pr, gvalid, fifo, gmode = lector.prox[-1]
            if time.time() < base_hasta:
                base.append(pr)
            else:
                pmax = max(pmax, pr)
            estado = "MOTOR ACTIVO" if gmode else ("cerca" if pr >= UMBRAL_PROX else "")
            print(f"\r  prox {pr:3d} {barra(pr)}  max {pmax:3d}  {estado:<13}",
                  end="", flush=True)
        if any("PROX fin" in o for o in lector.otras):
            break
        time.sleep(0.05)
    print()
    print()
    if base:
        print(f"  linea base sin mano: mediana {sorted(base)[len(base) // 2]}")
    print()

    # Tres desenlaces, no dos.
    #
    # La primera version solo distinguia "cruza el umbral" de "no lo cruza", y
    # eso mete en el mismo saco dos fallos con arreglos opuestos: un sensor que
    # no ve nada, y uno que te ve perfectamente pero con el umbral del motor de
    # gestos puesto demasiado alto. El segundo se arregla bajando un numero.
    if pmax >= UMBRAL_PROX:
        print(f"  SI TE VE. Proximidad maxima {pmax}, por encima del umbral de")
        print(f"  entrada ({UMBRAL_PROX}): el motor de gestos arranca.")
        return True

    if pmax >= 12:
        print(f"  TE VE, PERO POR DEBAJO DEL UMBRAL. Maxima {pmax}, hace falta "
              f"{UMBRAL_PROX}.")
        print()
        print("  El sensor detecta la mano, o sea que ni la distancia ni la")
        print("  orientacion son el problema. Lo que sobra es el umbral de")
        print("  entrada del motor de gestos.")
        print()
        print("  Prueba primero acercando mas la mano, a unos 5 cm. Si asi pasa")
        print(f"  de {UMBRAL_PROX}, con bajar el umbral queda resuelto: pasame")
        print("  este maximo y lo ajusto en el firmware.")
        return False

    print(f"  NO TE VE. Proximidad maxima {pmax}, practicamente la linea base.")
    print()
    print("  Ni siquiera detecta que hay una mano, asi que el motor de gestos no")
    print("  llega a arrancar y probar direcciones no diria nada. Por orden:")
    print("    1. .Esta la lente del APDS despejada y mirando hacia ti? Si quedo")
    print("       en medio del apilado o mirando hacia abajo, tiene el emisor")
    print("       infrarrojo tapado y ninguna ganancia lo arregla.")
    print("    2. .Acercaste la mano a menos de 10 cm?")
    print("    3. Si el maximo se queda en 0 exacto, el sensor no esta midiendo")
    print("       y el problema es de configuracion, no de distancia.")
    return False


def fase_direcciones(ser, lector: Lector) -> list[tuple[str, str]]:
    print()
    print("=" * 68)
    print("  FASE 2 · ¿Acierta la dirección?")
    print("=" * 68)
    print()
    print("  Un gesto cada vez, franco y continuo, a unos 10-15 cm.")
    print(f"  {REPETICIONES} repeticiones de cada dirección.")
    print()

    resultados: list[tuple[str, str]] = []
    for pedido, como, para_que in DIRECCIONES:
        print(f"  --- {pedido}  ({para_que}) ---")
        for r in range(REPETICIONES):
            lector.drena()
            lector.gestos.clear()
            print(f"    {r + 1}/{REPETICIONES}  pasa la mano {como} ... ",
                  end="", flush=True)
            t0 = time.time()
            visto = None
            while time.time() - t0 < ESPERA_GESTO:
                lector.drena()
                if lector.gestos:
                    visto = lector.gestos.pop(0)
                    break
                time.sleep(0.05)
            if visto is None:
                print("nada detectado")
                resultados.append((pedido, "NADA"))
            else:
                marca = "OK" if visto == pedido else f"<- salió {visto}"
                print(f"{visto:<10} {marca}")
                resultados.append((pedido, visto))
            time.sleep(1.6)   # margen para el antirrebote del firmware
        print()
    return resultados


def informe(resultados: list[tuple[str, str]]) -> int:
    print("=" * 68)
    print("  RESULTADO")
    print("=" * 68)
    print()
    nombres = [d[0] for d in DIRECCIONES]
    print(f"  {'pedido':<12}" + "".join(f"{n[:5]:>7}" for n in nombres) + f"{'nada':>7}")
    print("  " + "-" * (12 + 7 * (len(nombres) + 1)))
    aciertos = total = 0
    for pedido in nombres:
        fila = [r for p, r in resultados if p == pedido]
        cuenta = [sum(1 for x in fila if x == n) for n in nombres]
        nada = sum(1 for x in fila if x == "NADA")
        aciertos += cuenta[nombres.index(pedido)]
        total += len(fila)
        print(f"  {pedido:<12}" + "".join(f"{c:>7}" for c in cuenta) + f"{nada:>7}")
    print()

    if total == 0:
        print("  Sin datos.")
        return 1
    tasa = aciertos / total
    print(f"  aciertos: {aciertos}/{total} = {tasa:.0%}")
    print()

    detectados = sum(1 for _, r in resultados if r != "NADA")
    if detectados == 0:
        print("  No se detectó ningún gesto pese a que el sensor sí ve la mano.")
        print("  El fallo está en el decodificador o en el umbral de decisión,")
        print("  no en el sensor. Pásame esta salida.")
        return 1

    if tasa >= 0.8:
        print("  Los gestos funcionan y el mapeo es correcto.")
        return 0

    # ¿Hay una rotación constante? Se ve si cada dirección sale siempre como
    # otra fija: entonces no es ruido, es que los ejes están girados.
    mapa = {}
    for pedido in nombres:
        fila = [r for p, r in resultados if p == pedido and r != "NADA"]
        if fila:
            mapa[pedido] = max(set(fila), key=fila.count)
    consistente = len(set(mapa.values())) == len(mapa) and mapa and \
        all(v != k for k, v in mapa.items())
    print("  El mapeo NO coincide.")
    if consistente:
        print("  Pero es consistente, así que son los ejes del sensor girados")
        print("  respecto a cómo quedó montada la placa. Se corrige en el")
        print("  firmware, no repitiendo el gesto:")
        for k, v in mapa.items():
            print(f"    lo que llamas {k:<10} el sensor lo lee como {v}")
    else:
        print("  Y no es consistente, así que es ruido de detección: el gesto")
        print("  probablemente va demasiado rápido, demasiado lejos, o no cruza")
        print("  entero el campo del sensor.")
    print()
    print("  Pásame esta salida y lo ajusto.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prueba guiada de los gestos del APDS9960.")
    ap.add_argument("--puerto", default="COM9")
    ap.add_argument("--solo-proximidad", action="store_true")
    args = ap.parse_args()

    print()
    print("  PneumoCoach · prueba de gestos")
    print("  Solo hace falta la placa conectada. No te pongas nada.")
    print()

    ser = abrir(args.puerto)
    lector = Lector(ser)
    try:
        # Modo prueba: los gestos se detectan y se informan, pero NO avanzan la
        # sesion. Sin esto, el primer gesto hacia arriba arrancaria la
        # calibracion y los siguientes caerian en una fase que los ignora.
        ser.write(b"y")
        time.sleep(0.5)
        lector.drena()
        for o in lector.otras[-4:]:
            print(f"  {o}")
        print()

        if not fase_proximidad(ser, lector):
            return 1
        if args.solo_proximidad:
            return 0
        resultados = fase_direcciones(ser, lector)
        return informe(resultados)
    finally:
        try:
            ser.write(b"y")   # salir del modo prueba
            time.sleep(0.3)
        except Exception:
            pass
        ser.close()


if __name__ == "__main__":
    sys.exit(main())
