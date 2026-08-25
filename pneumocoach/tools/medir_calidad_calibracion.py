"""¿Puede el dispositivo saber si una calibración salió bien?

    python tools/medir_calidad_calibracion.py

Respuesta corta con los datos de hoy: no.

Por qué se pregunta
-------------------
La calibración por sesión pide al paciente dos maniobras de referencia y define
el eje `z = (x - dia) / (tor - dia)`. Si el paciente ejecuta las dos casi igual
-porque no distingue las técnicas, porque el sensor se movió, porque no entendió
la consigna- ese eje es ruido, y todo lo que venga después es un veredicto
inventado con aspecto de medición.

Sería muy útil que el dispositivo lo detectara solo y pidiera repetir. Este
script comprueba si alguno de los dos estadísticos candidatos puede hacerlo.

Cómo se mide
------------
  SEÑAL: bloque diafragmático contra bloque torácico de la misma sesión. Es una
         calibración que sí separó.
  NULO:  dos mitades del MISMO bloque. El paciente hizo lo mismo dos veces, que
         es exactamente el fallo que se quiere detectar.

Un estadístico sirve como puerta solo si su valor mínimo sobre la señal queda
por encima de su valor máximo sobre el nulo. Si los rangos se solapan, no existe
ningún umbral que acepte toda calibración buena y rechace alguna mala.

Qué salió
---------
Ninguno de los dos separa. El contraste relativo -que es adimensional, así que
el ruido tiene tanta variación relativa como una maniobra real- solapa mucho, y
la d de Cohen solapa menos pero solapa.

No es un fallo del estadístico: es la misma limitación que documenta ADR-0007.
La deriva dentro de una maniobra es del mismo orden que la diferencia entre
maniobras, así que no hay nada que separar. Mientras eso siga siendo cierto, el
dispositivo enseña el número como diagnóstico y no lo usa para certificar nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))


import _consola  # noqa: E402,F401  UTF-8 en Windows
from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402
from analizar_captura import bloques, cargar, cargar_calibracion  # noqa: E402
from medir_calibracion import ES_DIA, ES_TOR  # noqa: E402

# Tantas ventanas como recogería el dispositivo en una maniobra de referencia.
N_VENTANAS = 6


def ventanas(ch, a: int, b: int, n_max: int = N_VENTANAS) -> np.ndarray:
    X = []
    for i in range(a, b - C.WINDOW_N + 1, C.HOP_N):
        X.append(dsp.extract_features({k: v[i:i + C.WINDOW_N] for k, v in ch.items()}))
        if len(X) >= n_max:
            break
    return np.asarray(X)


def contraste_relativo(A: np.ndarray, B: np.ndarray) -> float:
    """Mediana de |tor - dia| / max(|dia|, |tor|). Adimensional."""
    a, b = A.mean(0), B.mean(0)
    escala = np.maximum(np.abs(a), np.abs(b)) + 1e-12
    return float(np.median(np.abs(b - a) / escala))


def d_de_cohen(A: np.ndarray, B: np.ndarray) -> float:
    """Mediana de la separación entre medias en unidades de la dispersión interna.

    Es el estadístico correcto en principio: pregunta si las dos maniobras se
    separan MÁS de lo que cada una varía por dentro. El contraste relativo no
    mira la dispersión interna en absoluto, y por eso el ruido lo engaña.
    """
    a, b = A.mean(0), B.mean(0)
    sp = np.sqrt((A.var(0, ddof=1) + B.var(0, ddof=1)) / 2) + 1e-12
    return float(np.median(np.abs(b - a) / sp))


ESTADISTICOS = (
    ("contraste relativo", contraste_relativo),
    ("d de Cohen", d_de_cohen),
)


def main() -> int:
    R, sesgo = cargar_calibracion("s01")
    csvs = sorted((REPO / "data" / "raw").glob("*.csv"))
    if not csvs:
        print("No hay grabaciones en data/raw/.")
        return 1

    senal: dict[str, list[float]] = {n: [] for n, _ in ESTADISTICOS}
    nulo: dict[str, list[float]] = {n: [] for n, _ in ESTADISTICOS}

    for csv in csvs:
        counts, et = cargar(csv)
        ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)
        bl = [(t, a, b) for t, a, b in bloques(et)
              if t in ES_DIA + ES_TOR and b - a >= 2 * C.WINDOW_N]

        for tec_a, a0, a1 in (x for x in bl if x[0] in ES_DIA):
            for tec_b, b0, b1 in (x for x in bl if x[0] in ES_TOR):
                A, B = ventanas(ch, a0, a1), ventanas(ch, b0, b1)
                if len(A) > 1 and len(B) > 1:
                    for n, f in ESTADISTICOS:
                        senal[n].append(f(A, B))

        # Nulo: dos mitades del mismo bloque.
        for tec, a, b in bl:
            m = a + (b - a) // 2
            A, B = ventanas(ch, a, m), ventanas(ch, m, b)
            if len(A) > 1 and len(B) > 1:
                for n, f in ESTADISTICOS:
                    nulo[n].append(f(A, B))

    print(f"  {len(csvs)} grabaciones, {N_VENTANAS} ventanas por maniobra\n")
    separa_alguno = False
    for nombre, _ in ESTADISTICOS:
        s_ = np.asarray(senal[nombre])
        n_ = np.asarray(nulo[nombre])
        if not len(s_) or not len(n_):
            continue
        separa = s_.min() > n_.max()
        separa_alguno |= separa
        print(f"  {nombre}")
        print(f"    SENAL (n={len(s_):2d})  min {s_.min():.3f}  "
              f"mediana {np.median(s_):.3f}  max {s_.max():.3f}")
        print(f"    NULO  (n={len(n_):2d})  min {n_.min():.3f}  "
              f"mediana {np.median(n_):.3f}  max {n_.max():.3f}")
        if separa:
            print(f"    SEPARA. Umbral utilizable entre {n_.max():.3f} y {s_.min():.3f}.")
        else:
            print(f"    NO SEPARA: los rangos se solapan. Cualquier umbral que")
            print(f"    acepte toda la senal acepta tambien parte del nulo.")
        print()

    print("=" * 70)
    if separa_alguno:
        print("  Hay al menos un estadistico con umbral defendible. Se puede")
        print("  poner la puerta de calidad en el dispositivo.")
        return 0
    print("  Ningun estadistico separa. El dispositivo NO puede certificar que")
    print("  una calibracion sirva, y no debe pretenderlo: ensena el numero como")
    print("  diagnostico y nada mas.")
    print()
    print("  Es la misma limitacion de ADR-0007. La deriva dentro de una maniobra")
    print("  es del mismo orden que la diferencia entre maniobras; hasta que haya")
    print("  una referencia independiente que las etiquete, no hay nada que")
    print("  separar y ninguna puerta puede inventarselo.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
