"""Cuanto tarda la cadena de filtros en dar una salida fiable.

    python tools/medir_asentamiento.py

De aqui sale WARMUP_S. Antes era 60 s, elegido a ojo; medido son ~12 s.

Metodo
------
Se filtra la grabacion dos veces: una empezando en la muestra 0, y otra tras
anteponer 30 s de la propia senal invertida en el tiempo. La segunda hace las
veces de "filtro que ya lleva rato funcionando" sin inventar datos que no
existen. La distancia entre ambas salidas es el error que arrastra el arranque;
el asentamiento es el instante tras el cual esa distancia se queda por debajo
del 1 % del RMS de la senal.

El criterio es "se queda por debajo", no "baja por primera vez": un transitorio
que cruza el umbral y vuelve a salir no esta asentado.
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
from analizar_captura import cargar  # noqa: E402

PREFIJO_N = 1500  # 30 s de historia sintetica
UMBRAL_REL = 0.01
CANALES = ("tilt", "axial", "hf")


def asentamiento(y: np.ndarray, y_ref: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(y_ref[dsp.WARMUP_N:] ** 2)))
    if rms <= 0.0:
        return float("nan")
    bajo = np.abs(y - y_ref) < UMBRAL_REL * rms
    idx = next((i for i in range(len(bajo)) if bajo[i:].all()), len(bajo) - 1)
    return idx / C.FS_HZ


def main() -> int:
    csvs = sorted((REPO / "data" / "raw").glob("*.csv"))
    if not csvs:
        print("No hay grabaciones en data/raw/. Nada que medir.")
        return 1

    print(f"{'grabacion':<34}" + "".join(f"{c:>9}" for c in CANALES))
    print("-" * (34 + 9 * len(CANALES)))
    peor = {c: 0.0 for c in CANALES}
    for csv in csvs:
        counts, _ = cargar(csv)
        if len(counts) < PREFIJO_N * 2:
            continue
        ch = dsp.channels_from_counts(counts)
        pre = counts[:PREFIJO_N][::-1]
        ch_ref = dsp.channels_from_counts(np.vstack([pre, counts]))
        fila = []
        for c in CANALES:
            t = asentamiento(ch[c], ch_ref[c][PREFIJO_N:])
            peor[c] = max(peor[c], t)
            fila.append(f"{t:>8.1f}s")
        print(f"{csv.stem[:33]:<34}" + "".join(fila))

    print("-" * (34 + 9 * len(CANALES)))
    print(f"{'PEOR CASO':<34}" + "".join(f"{peor[c]:>8.1f}s" for c in CANALES))
    p = max(peor.values())
    print()
    print(f"  Peor asentamiento medido: {p:.1f} s")
    print(f"  WARMUP_S configurado:     {dsp.WARMUP_S:.1f} s "
          f"(margen x{dsp.WARMUP_S / p:.2f})")
    if dsp.WARMUP_S < p:
        print("  WARMUP_S ES MENOR QUE EL ASENTAMIENTO MEDIDO. Las primeras")
        print("  ventanas entran al modelo con el transitorio dentro.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
