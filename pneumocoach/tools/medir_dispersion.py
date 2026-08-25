"""Dispersion real de cada caracteristica sobre las grabaciones disponibles.

    python tools/medir_dispersion.py

De aqui salen los pisos absolutos de tools/paridad.py. La regla es 0.004 sigma:
por debajo de esa diferencia dos implementaciones no pueden discrepar en ningun
veredicto, asi que exigir mas es exigir ruido.

Solo hace falta para las caracteristicas cuyo valor pasa por cero -una
correlacion, un coeficiente de variacion casi nulo-, donde el criterio relativo
deja de significar nada. Para el resto manda rtol y este piso no llega a actuar.
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

FRACCION = 0.004


def main() -> int:
    csvs = sorted((REPO / "data" / "raw").glob("*.csv"))
    if not csvs:
        print("No hay grabaciones en data/raw/.")
        return 1

    X = []
    for csv in csvs:
        counts, _ = cargar(csv)
        ch = dsp.channels_from_counts(counts)
        for ini in range(0, len(counts) - C.WINDOW_N + 1, C.HOP_N):
            if ini < dsp.WARMUP_N:
                continue
            w = {k: v[ini:ini + C.WINDOW_N] for k, v in ch.items()}
            X.append(dsp.extract_features(w))
    X = np.asarray(X)

    print(f"  {len(X)} ventanas de {len(csvs)} grabaciones\n")
    print(f"  {'caracteristica':<24}{'sigma':>11}{'|media|':>11}"
          f"{f'{FRACCION} sigma':>13}")
    print("  " + "-" * 57)
    for j, nombre in enumerate(C.FEATURE_NAMES):
        sd, mu = X[:, j].std(), abs(X[:, j].mean())
        # Marcamos las que rondan el cero: son las que necesitan piso absoluto.
        marca = "  <- pasa por cero" if mu < sd else ""
        print(f"  {nombre:<24}{sd:>11.4f}{mu:>11.4f}{FRACCION * sd:>13.2e}{marca}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
