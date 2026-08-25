"""Comprueba que la sesión se completa sola, sin ninguna orden.

    python tools/verificar_sesion_auto.py --puerto COM9

No hace falta ponerse el sensor ni tocar nada: la placa sobre la mesa basta.
La herramienta se limita a mirar; no envía ni una sola transición.

Por qué existe
--------------
El dispositivo tiene que poder dirigir una sesión clínica sin PC y sin que el
paciente accione nada (ADR-0010). Eso es una afirmación sobre el comportamiento
del firmware, y como toda afirmación de este proyecto necesita una comprobación
que se pueda repetir.

Verifica dos cosas distintas:

  LA SECUENCIA. Que pasa por las seis fases en orden.
  LOS TIEMPOS.  Que cada fase dura lo que dice el contrato compartido. Una
                secuencia correcta con una maniobra de 12 s en vez de 30 no
                recogería suficientes ventanas de referencia, y desde fuera se
                vería igual de bien.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401
from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402
from paridad import abrir  # noqa: E402

# Fase, duración esperada en segundos, tolerancia.
#
# La tolerancia es de 2 s: el estado se consulta una vez por segundo y la fase
# avanza por cuenta de muestras, así que un segundo de desfase es del muestreo,
# no del firmware.
ESPERADO = [
    ("ASENTANDO",    dsp.WARMUP_S,          2.0),
    ("LISTO",        C.PREPARA_SEGUNDOS,    2.0),
    ("CAL DIAFRAG",  C.REF_SEGUNDOS,        2.0),
    ("CAL TORACICA", C.REF_SEGUNDOS,        2.0),
    ("CALIBRADO",    C.RESULTADO_SEGUNDOS,  2.0),
    ("SESION",       None,                  0.0),   # final, no expira
]


def fase_de(linea: str) -> str:
    """Extrae la fase de '# SESION <FASE>  n=...'.

    Trocear por 'SESION ' no vale: la fase final se llama SESION, asi que la
    cadena aparece dos veces y el troceo devuelve vacio. Es el fallo que hizo
    que una sesion correcta se reportara como fallida.
    """
    marca = "# SESION "
    if not linea.startswith(marca):
        return ""
    return linea[len(marca):].split("  ")[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", default="COM9")
    args = ap.parse_args()

    limite = sum(d for _, d, _ in ESPERADO if d) + 30.0
    print("=" * 70)
    print("  SESION AUTOMATICA  ·  la herramienta solo mira, no ordena nada")
    print("=" * 70)
    print(f"\n  duracion esperada ~{limite - 30:.0f} s\n")

    ser = abrir(args.puerto)
    t0 = time.time()
    transiciones: list[tuple[str, float]] = []
    previa = ""
    try:
        while time.time() - t0 < limite:
            ser.reset_input_buffer()
            ser.write(b"c")
            time.sleep(0.35)
            linea = ser.read(ser.in_waiting or 0).decode("utf-8", "replace").strip()
            f = fase_de(linea)
            if f and f != previa:
                t = time.time() - t0
                transiciones.append((f, t))
                print(f"  [{t:6.1f}s] {f}")
                previa = f
                if f == "SESION":
                    break
            time.sleep(0.7)
    finally:
        ser.close()

    print()
    print("=" * 70)
    vistas = [f for f, _ in transiciones]
    esperadas = [f for f, _, _ in ESPERADO]
    if vistas != esperadas:
        print("  SECUENCIA INCORRECTA")
        print(f"    vista:    {' -> '.join(vistas)}")
        print(f"    esperada: {' -> '.join(esperadas)}")
        return 1

    print("  SECUENCIA CORRECTA. Duraciones:")
    print()
    print(f"  {'fase':<16}{'medido':>9}{'esperado':>10}{'':>4}")
    mal = 0
    for i, (nombre, dur, tol) in enumerate(ESPERADO[:-1]):
        medido = transiciones[i + 1][1] - transiciones[i][1]
        ok = abs(medido - dur) <= tol
        if not ok:
            mal += 1
        print(f"  {nombre:<16}{medido:>8.1f}s{dur:>9.0f}s   {'ok' if ok else '<- FUERA'}")
    print()
    if mal:
        print(f"  {mal} fases fuera de tolerancia. La secuencia es correcta pero")
        print("  los tiempos no, y de ellos depende cuantas ventanas de")
        print("  referencia se recogen.")
        return 1
    print("  La sesion se completa sola, en orden y en tiempo.")
    print("  Sin ninguna orden enviada, sin PC y sin que el paciente toque nada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
