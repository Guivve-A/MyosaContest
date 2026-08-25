"""La compuerta: .describe la consigna lo que hizo el cuerpo?

    python tools/compuerta_etiquetas.py --autoprueba
    python tools/compuerta_etiquetas.py data/etiquetas/*_respiraciones.csv

REGISTRO PREVIO
===============
Este fichero se escribio el 23 de agosto de 2026, ANTES de que existiera
ninguna sesion con etiqueta medida. Eso no es una anecdota: es lo que separa
una prueba de hipotesis de una racionalizacion. Si algo de lo de abajo cambia
despues de ver los datos, el cambio se anota aqui con su fecha y su motivo.

Hipotesis primaria
------------------
    H0: el indice de contribucion toracica de los bloques `diaphragmatic` NO es
        menor que el de los bloques `thoracic`.
    H1: si lo es.

    Prueba   : Mann-Whitney U, unilateral
    Alfa     : 0.05
    Unidad   : LA RESPIRACION. Nunca la ventana solapada.
    Efecto   : delta de Cliff, con IC del 95 % por bootstrap sobre respiraciones

Por que Mann-Whitney y no una t: `rc_ix` es un logaritmo de un cociente de
amplitudes y no hay ninguna razon para esperarlo gaussiano. Y por que delta de
Cliff y no d de Cohen: delta es una probabilidad de orden, no depende de la
escala, y este proyecto ya tiene dos convenios de signo incompatibles para d
(`analizar_captura.py` resta al reves que `diagnostico_premisa.py`).

Por que la unidad es la respiracion
-----------------------------------
Las ventanas de 12 s con salto de 3 s comparten el 75 % de sus muestras.
Contarlas como independientes multiplica el n aparente por cuatro y estrecha
todos los intervalos por dos. Es el error que inflo dos cifras de este
proyecto, las dos veces sin que nadie lo notara hasta comparar sesiones.

Que se decide con el resultado
------------------------------
Los dos caminos estan escritos ANTES de mirar, para que el dato no elija el
analisis:

  PASA  -> la consigna describe la mecanica para este sujeto. Las etiquetas de
           las 4 sesiones anteriores valen, y el protocolo de dos manos quedo
           validado contra una medicion.

  FALLA -> se reetiqueta por `rc_ix` medido y se reentrena. El resultado
           publicable pasa a ser "el N % de los bloques estaban mal
           etiquetados", que es la validacion empirica de ADR-0007 y vale mas
           que una cifra de exactitud.

En los dos casos hay resultado. Ninguno es un fracaso.
"""

from __future__ import annotations

import argparse
import csv as _csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401

ALFA = 0.05
N_BOOTSTRAP = 10000

ES_DIA = ("diaphragmatic", "dia_suave", "dia_fuerte")
ES_TOR = ("thoracic", "tor_suave", "tor_fuerte")


def cliff(a: np.ndarray, b: np.ndarray) -> float:
    """Delta de Cliff: P(a > b) - P(a < b), en [-1, 1].

    Probabilidad de orden. No supone ninguna distribucion y no depende de la
    escala, asi que sobrevive a que `rc_ix` tenga un cero arbitrario.
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # Via el estadistico U, que es O(n log n) en vez de O(n^2).
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def ic_cliff(a: np.ndarray, b: np.ndarray, semilla: int = 0) -> tuple[float, float]:
    """IC 95 % de delta por bootstrap sobre respiraciones."""
    if len(a) < 3 or len(b) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(semilla)
    ds = [cliff(rng.choice(a, len(a), replace=True),
                rng.choice(b, len(b), replace=True))
          for _ in range(N_BOOTSTRAP)]
    lo, hi = np.percentile([d for d in ds if np.isfinite(d)], [2.5, 97.5])
    return float(lo), float(hi)


def carga(rutas: list[str]) -> dict[str, list[dict]]:
    """Respiraciones validas por sesion."""
    por_sesion: dict[str, list[dict]] = {}
    for r in rutas:
        p = Path(r)
        filas = []
        with p.open(encoding="utf-8") as fh:
            for f in _csv.DictReader(fh):
                if f.get("valido", "").lower() not in ("true", "1"):
                    continue
                try:
                    filas.append({
                        "consigna": f["consigna"],
                        "rc_ix": float(f["rc_ix"]),
                        "phi": float(f["phi_grados"]),
                        "rpm": float(f["frecuencia_rpm"]),
                        "amp": float(f["amplitud_mareal"]),
                    })
                except (KeyError, ValueError):
                    continue
        if filas:
            por_sesion[p.stem.replace("_respiraciones", "")] = filas
    return por_sesion


def informe(por_sesion: dict[str, list[dict]]) -> bool:
    print("=" * 78)
    print("  COMPUERTA: .DESCRIBE LA CONSIGNA LO QUE HIZO EL CUERPO?")
    print("=" * 78)
    print()
    print("  Hipotesis registrada antes de grabar: rc_ix(diafragmatica) <")
    print(f"  rc_ix(toracica). Mann-Whitney unilateral, alfa = {ALFA}.")
    print("  Unidad: la respiracion.")
    print()
    print(f"  {'sesion':<28}{'n dia':>7}{'n tor':>7}{'delta':>8}"
          f"{'IC 95 %':>20}{'p':>11}")
    print("  " + "-" * 76)

    pasa_todas = True
    for nombre, filas in sorted(por_sesion.items()):
        dia = np.array([f["rc_ix"] for f in filas if f["consigna"] in ES_DIA])
        tor = np.array([f["rc_ix"] for f in filas if f["consigna"] in ES_TOR])
        if len(dia) < 3 or len(tor) < 3:
            print(f"  {nombre:<28}{len(dia):>7}{len(tor):>7}"
                  f"{'  sin datos suficientes':>39}")
            pasa_todas = False
            continue
        p = stats.mannwhitneyu(dia, tor, alternative="less").pvalue
        d = cliff(dia, tor)
        lo, hi = ic_cliff(dia, tor)
        marca = "" if p < ALFA else "  <- NO"
        pasa_todas &= p < ALFA
        print(f"  {nombre:<28}{len(dia):>7}{len(tor):>7}{d:>8.2f}"
              f"   [{lo:+.2f}, {hi:+.2f}]{p:>11.2e}{marca}")

    print("  " + "-" * 76)
    print()

    # --- concordancia bloque a bloque -------------------------------------
    # La pregunta de ADR-0007 en su forma mas directa: .cuantos bloques hicieron
    # mecanicamente lo contrario de lo que decia su consigna?
    print("  CONCORDANCIA. Un bloque discrepa si su rc_ix mediano cae del lado")
    print("  equivocado de la mediana global de la sesion.")
    print()
    total = discrepan = 0
    for nombre, filas in sorted(por_sesion.items()):
        mediana = float(np.median([f["rc_ix"] for f in filas]))
        por_consigna: dict[str, list[float]] = defaultdict(list)
        for f in filas:
            por_consigna[f["consigna"]].append(f["rc_ix"])
        for consigna, vals in sorted(por_consigna.items()):
            if consigna not in ES_DIA + ES_TOR:
                continue
            m = float(np.median(vals))
            esperado_alto = consigna in ES_TOR
            mal = (m > mediana) != esperado_alto
            total += 1
            discrepan += mal
            print(f"    {nombre[:22]:<24}{consigna:<16}"
                  f"mediana {m:+.3f}  {'DISCREPA' if mal else 'concuerda'}")
    print()
    if total:
        print(f"  Bloques que contradicen su consigna: {discrepan} de {total} "
              f"({discrepan / total:.0%})")
    print()

    veredicto = pasa_todas and total and discrepan == 0
    print("=" * 78)
    if veredicto:
        print("  LA COMPUERTA PASA.")
        print("  La consigna describe la mecanica para este sujeto. Las")
        print("  etiquetas anteriores valen y el protocolo de dos manos queda")
        print("  validado contra una medicion independiente.")
    else:
        print("  LA COMPUERTA NO PASA.")
        print("  Segun lo registrado de antemano: reetiquetar por rc_ix medido")
        print("  y reentrenar. El resultado que se publica es la fraccion de")
        print("  bloques mal etiquetados, no una exactitud.")
    print("=" * 78)
    return veredicto


def autoprueba() -> int:
    """La prueba tiene que detectar una separacion real y NO inventarla."""
    print("=" * 78)
    print("  AUTOPRUEBA DE LA COMPUERTA")
    print("=" * 78)
    print()
    rng = np.random.default_rng(11)
    ok = True

    # Caso 1: separacion real. Tiene que salir significativa y delta negativa.
    dia = rng.normal(-0.4, 0.25, 40)
    tor = rng.normal(+0.4, 0.25, 40)
    p = stats.mannwhitneyu(dia, tor, alternative="less").pvalue
    d = cliff(dia, tor)
    lo, hi = ic_cliff(dia, tor)
    bien = p < ALFA and d < -0.5 and hi < 0
    ok &= bien
    print(f"  separacion real     delta {d:+.2f}  IC [{lo:+.2f}, {hi:+.2f}]"
          f"  p = {p:.1e}  {'ok' if bien else 'FALLA'}")

    # Caso 2: sin separacion. NO debe salir significativa, y el IC debe cruzar 0.
    a = rng.normal(0.0, 0.3, 40)
    b = rng.normal(0.0, 0.3, 40)
    p = stats.mannwhitneyu(a, b, alternative="less").pvalue
    d = cliff(a, b)
    lo, hi = ic_cliff(a, b)
    bien = p > ALFA and lo < 0 < hi
    ok &= bien
    print(f"  sin separacion      delta {d:+.2f}  IC [{lo:+.2f}, {hi:+.2f}]"
          f"  p = {p:.2f}  {'ok' if bien else 'FALLA'}")

    # Caso 3: separacion INVERTIDA -el caso de ADR-0007-. La prueba unilateral
    # NO debe declararla significativa: buscamos dia < tor, no "hay diferencia".
    dia = rng.normal(+0.4, 0.25, 40)
    tor = rng.normal(-0.4, 0.25, 40)
    p = stats.mannwhitneyu(dia, tor, alternative="less").pvalue
    d = cliff(dia, tor)
    bien = p > ALFA and d > 0.5
    ok &= bien
    print(f"  separacion INVERSA  delta {d:+.2f}  p = {p:.2f}"
          f"  {'ok' if bien else 'FALLA'}"
          f"   (unilateral: no la declara significativa)")

    print()
    print("  AUTOPRUEBA " + ("PASA" if ok else "FALLA"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="*", help="CSV de respiraciones")
    ap.add_argument("--autoprueba", action="store_true")
    args = ap.parse_args()

    if args.autoprueba:
        return autoprueba()

    rutas = args.csv or sorted(
        glob.glob(str(REPO / "data" / "etiquetas" / "*_respiraciones.csv")))
    if not rutas:
        raise SystemExit(
            "No hay sesiones con etiqueta medida todavia.\n"
            "  1. Grabar segun docs/DUAL_SENSOR_PROTOCOL.md\n"
            "  2. python tools/capturar_dual.py --alinear ... --telefono ...\n"
            "  3. python tools/etiqueta_objetiva.py ... --abdomen ...")

    por_sesion = carga(rutas)
    if not por_sesion:
        raise SystemExit("Los CSV no traen respiraciones validas.")
    return 0 if informe(por_sesion) else 2


if __name__ == "__main__":
    sys.exit(main())
