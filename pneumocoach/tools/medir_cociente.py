"""Que hace de verdad el cociente inclinacion/axial, sesion por sesion.

    python tools/medir_cociente.py

Por que hace falta
------------------
Dos documentos del repositorio daban dos cifras distintas para lo mismo:
"d = -1.44 en la mejor sesion, negativo en las 4" en las notas tecnicas, y
"d = -0.05 sobre el primer pecho real" en el blog. No se contradicen -son datos
distintos- pero cualquiera que cite una sin la otra afirma de mas.

La app y el blog van a describir este cociente en texto que lee un jurado, asi
que la descripcion sale de aqui y no de acordarse.

Que se mide
-----------
`log_tilt_axial_ratio`, la caracteristica 20, entre ventanas diafragmaticas y
toracicas. Cohen's d con desviacion agrupada, y el signo IMPORTA: la premisa
del diseno predecia d POSITIVO -mas cociente en toracica- y ADR-0006 lo
encontro al reves.

Se mide sobre la caracteristica ABSOLUTA, sin proyectar al eje del paciente,
porque eso es lo que la app muestra en pantalla.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "ml" / "scripts"))
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401
from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402
from analizar_captura import bloques, cargar, cargar_calibracion  # noqa: E402
import entrenar_real as ER  # noqa: E402

I_RATIO = list(C.FEATURE_NAMES).index("log_tilt_axial_ratio")
I_TILT = list(C.FEATURE_NAMES).index("tilt_rms")
I_AXIAL = list(C.FEATURE_NAMES).index("axial_rms")


SOLAPE = int(C.WINDOW_N / C.HOP_N)     # ventanas que comparten datos: 4


def d_cohen(a: np.ndarray, b: np.ndarray) -> float:
    """d de a respecto de b, con desviacion agrupada.

    Convenio: a = TORACICA, b = DIAFRAGMATICA. La premisa original predecia
    POSITIVO. Lo que salga negativo es la premisa invertida.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    s = np.sqrt(((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2)
                / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")


def d_independiente(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """d sobre ventanas que NO comparten datos: (mediana, minimo, maximo).

    Con ventana de 12 s y salto de 3 s, cuatro ventanas consecutivas comparten
    el 75 % de sus muestras. Una d calculada sobre todas ellas cuenta cuatro
    veces cada respiracion y sale inflada, y el sesgo es peor cuanto MENOS dura
    la sesion — que es justo donde aparecieron las magnitudes grandes.

    Se toma una de cada cuatro ventanas. Como el resultado depende de por cual
    se empiece, se recorren los cuatro desfases y se informa el rango. Un
    efecto que no sobrevive a eso no era un efecto.
    """
    ds = [d_cohen(a[k::SOLAPE], b[k::SOLAPE]) for k in range(SOLAPE)]
    ds = [d for d in ds if np.isfinite(d)]
    if not ds:
        return (float("nan"),) * 3
    return float(np.median(ds)), float(min(ds)), float(max(ds))


def caracteristicas(path: str):
    p = REPO / path
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    R, sesgo = cargar_calibracion("s01", meta.get("inicio_utc"))
    counts, et = cargar(p)
    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)
    X, y = [], []
    for tec, a, b in bloques(et):
        if tec not in ER.ES_DIA + ER.ES_TOR:
            continue
        for i in range(a, b - C.WINDOW_N + 1, C.HOP_N):
            X.append(dsp.extract_features(
                {k: v[i:i + C.WINDOW_N] for k, v in ch.items()}))
            y.append(0 if tec in ER.ES_DIA else 1)
    return np.stack(X), np.asarray(y)


def main() -> int:
    print("=" * 74)
    print("  EL COCIENTE INCLINACION / AXIAL, SESION POR SESION")
    print("=" * 74)
    print()
    print("  La premisa del diseno predecia d POSITIVO (mas cociente en la")
    print("  toracica). Negativo = premisa invertida.")
    print()
    print(f"  {'sesion':<16}{'d solapada':>12}{'d independiente':>28}"
          f"{'n/clase':>9}{'n indep':>9}")
    print("  " + "-" * 72)

    todos_r, todas_et = [], []
    for nombre, (path, _) in ER.SESIONES.items():
        X, y = caracteristicas(path)
        tor, dia = X[y == 1], X[y == 0]
        med, lo, hi = d_independiente(tor[:, I_RATIO], dia[:, I_RATIO])
        print(f"  {nombre:<16}"
              f"{d_cohen(tor[:, I_RATIO], dia[:, I_RATIO]):>12.2f}"
              f"{med:>16.2f}  [{lo:+.2f}, {hi:+.2f}]"
              f"{len(dia):>9}{len(dia) // SOLAPE:>9}")
        todos_r.append(X[:, [I_RATIO, I_TILT, I_AXIAL]])
        todas_et.append(y)

    X = np.vstack(todos_r)
    y = np.concatenate(todas_et)
    tor, dia = X[y == 1], X[y == 0]
    med, lo, hi = d_independiente(tor[:, 0], dia[:, 0])
    print("  " + "-" * 72)
    print(f"  {'las 4 juntas':<16}{d_cohen(tor[:, 0], dia[:, 0]):>12.2f}"
          f"{med:>16.2f}  [{lo:+.2f}, {hi:+.2f}]"
          f"{len(dia):>9}{len(dia) // SOLAPE:>9}")
    print()
    print("  Como leerlo:")
    print("  - El signo del cociente es lo que ADR-0006 refuto. Comprobar que")
    print("    sigue siendo el mismo antes de escribirlo en la app o el blog.")
    print("  - Los dos canales por separado pueden separar bien y aun asi el")
    print("    cociente no separar: si suben JUNTOS, la division los cancela.")
    print()
    estabilidad()
    return 0


def estabilidad() -> None:
    """Cuantas de las 29 mantienen el SIGNO de su efecto entre sesiones.

    El blog afirma que `axial_rms` es "the only one of the 29 features that
    keeps the sign of its effect". Si eso es falso es una afirmacion publicada
    sin herramienta detras, que es exactamente lo que este repositorio no
    admite. Se cuenta, no se recuerda.
    """
    print("=" * 74)
    print("  ESTABILIDAD DE SIGNO ENTRE SESIONES")
    print("=" * 74)
    print()
    por_sesion = []
    for _, (path, _) in ER.SESIONES.items():
        X, y = caracteristicas(path)
        tor, dia = X[y == 1], X[y == 0]
        por_sesion.append(np.array(
            [d_cohen(tor[:, i], dia[:, i]) for i in range(C.N_FEATURES)]))
    D = np.stack(por_sesion)          # (sesiones, caracteristicas)

    nombres = list(C.FEATURE_NAMES)
    for k, etiqueta in ((3, "las 3 primeras"), (4, "las 4")):
        sub = D[:k]
        mismo = np.all(np.sign(sub) == np.sign(sub[0]), axis=0)
        mismo &= ~np.any(np.isnan(sub), axis=0)
        print(f"  Mantienen el signo en {etiqueta}: "
              f"{int(mismo.sum())} de {C.N_FEATURES}")
        if k == 4:
            estables = [nombres[i] for i in np.where(mismo)[0]]
            print(f"    {', '.join(estables)}")
    print()
    print("  Las dos que decide el diseno:")
    for nombre in ("tilt_rms", "axial_rms"):
        i = nombres.index(nombre)
        fila = "  ".join(f"{v:+.2f}" for v in D[:, i])
        cambia = "CAMBIA DE SIGNO" if len(set(np.sign(D[:, i]))) > 1 else "estable"
        print(f"    {nombre:<12} {fila}   -> {cambia}")
    print()
    print("  Convenio en todo este fichero: d = (toracica - diafragmatica),")
    print("  el mismo que tools/analizar_captura.py. tools/diagnostico_premisa.py")
    print("  usa el CONTRARIO, y por eso el repositorio tiene dos signos para")
    print("  la misma medida. Citar una d sin decir el convenio no significa nada.")


if __name__ == "__main__":
    sys.exit(main())
