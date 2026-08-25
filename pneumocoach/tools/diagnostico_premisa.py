"""Por qué falló el discriminador, y qué separa realmente en datos reales.

    python tools/diagnostico_premisa.py data/raw/<captura>.csv

`analizar_captura.py` responde si la premisa se sostiene. Cuando la respuesta
es no, esto responde por qué, distinguiendo entre tres explicaciones que exigen
acciones muy distintas:

  A. El canal axial está en el piso de ruido. Entonces log(tilt/axial) no es un
     cociente entre dos señales sino un proxy de la amplitud de tilt, y nunca
     pudo discriminar técnica.
  B. El sujeto no produjo técnicas distinguibles. Una persona sin entrenamiento
     respiratorio a menudo no puede separarlas voluntariamente.
  C. La premisa mecánica es falsa: el esternón no rota diferencialmente.

Las tres se ven distinto en los datos y llevan a decisiones opuestas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))

from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402

sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401  UTF-8 en Windows
from analizar_captura import bloques, cargar, cargar_calibracion  # noqa: E402


def ventanas(counts, etiquetas, R, sesgo):
    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)
    X, y = [], []
    for tec, a, b in bloques(etiquetas):
        if b - a < C.WINDOW_N:
            continue
        for i in range(a, b - C.WINDOW_N + 1, C.HOP_N):
            X.append(dsp.extract_features({k: v[i:i + C.WINDOW_N] for k, v in ch.items()}))
            y.append(tec)
    return np.stack(X), np.asarray(y), ch


def cohen(a, b):
    pooled = np.sqrt(0.5 * (a.var() + b.var())) + 1e-12
    return (a.mean() - b.mean()) / pooled


def main():
    csv = Path(sys.argv[1])
    counts, etiquetas = cargar(csv)
    R, sesgo = cargar_calibracion(csv.stem.split("_")[0])
    X, y, ch = ventanas(counts, etiquetas, R, sesgo)

    dia, tor = X[y == "diaphragmatic"], X[y == "thoracic"]
    print("\n" + "=" * 70)
    print(f"  DIAGNÓSTICO · {len(dia)} ventanas diafragmática vs {len(tor)} torácica")
    print("=" * 70)

    # ---- Hipótesis A: ¿el canal axial es solo ruido? --------------------
    print("\n[A] ¿Está el canal axial en el piso de ruido?")
    j_ax = C.FEATURE_NAMES.index("axial_rms")
    j_ti = C.FEATURE_NAMES.index("tilt_rms")
    ax_all = X[:, j_ax]
    # Piso esperado: densidad medida sobre el ancho de banda del paso banda.
    piso = (C.ACCEL_NOISE_DENSITY_UG_RTHZ * 1e-6) * np.sqrt(C.BP_HIGH_HZ - C.BP_LOW_HZ)
    print(f"    axial_rms medido      {ax_all.mean() * 1000:7.3f} mg   "
          f"(desv {ax_all.std() * 1000:.3f})")
    print(f"    piso de ruido teórico {piso * 1000:7.3f} mg")
    print(f"    SNR aparente          {ax_all.mean() / piso:7.2f}")
    cv_ax = ax_all.std() / ax_all.mean()
    print(f"    coef. de variación    {cv_ax:7.3f}   "
          f"({'plano = ruido' if cv_ax < 0.35 else 'varía = hay señal'})")

    d_ax = cohen(dia[:, j_ax], tor[:, j_ax])
    d_ti = cohen(dia[:, j_ti], tor[:, j_ti])
    print(f"\n    Cohen's d (dia vs tor) sobre axial_rms  {d_ax:+.3f}")
    print(f"    Cohen's d (dia vs tor) sobre tilt_rms   {d_ti:+.3f}")
    if abs(d_ax) < 0.3 and abs(d_ti) < 0.3:
        print("    -> Ninguno de los dos canales separa por sí solo.")

    # ---- Hipótesis B: ¿produjo el sujeto técnicas distintas? ------------
    print("\n[B] ¿Produjo el sujeto dos técnicas distinguibles?")
    for nombre in ("breath_rate_bpm", "ie_ratio_mean", "tilt_p2p", "axial_p2p"):
        k = C.FEATURE_NAMES.index(nombre)
        print(f"    {nombre:<18} dia {dia[:, k].mean():8.3f}   "
              f"tor {tor[:, k].mean():8.3f}   d = {cohen(dia[:, k], tor[:, k]):+6.2f}")

    # ---- Qué separa realmente -------------------------------------------
    print("\n[C] Ranking empírico: qué separa diafragmática de torácica")
    ds = [(abs(cohen(dia[:, k], tor[:, k])), cohen(dia[:, k], tor[:, k]), n)
          for k, n in enumerate(C.FEATURE_NAMES)]
    ds.sort(reverse=True)
    print(f"    {'característica':<26}{'|d|':>7}{'signo':>8}")
    for mag, signo, n in ds[:10]:
        marca = "  <-- útil" if mag > 0.8 else ("  débil" if mag > 0.5 else "")
        print(f"    {n:<26}{mag:>7.2f}{signo:>+8.2f}{marca}")
    utiles = sum(1 for m, _, _ in ds if m > 0.8)
    print(f"\n    {utiles} de {len(ds)} características superan |d| = 0.8")

    # ---- ¿Puede un clasificador separarlas? -----------------------------
    print("\n[D] ¿Puede un clasificador separarlas, usando TODAS las características?")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    Xb = np.vstack([dia, tor])
    yb = np.r_[np.zeros(len(dia)), np.ones(len(tor))]
    Xs = StandardScaler().fit_transform(Xb)
    # Aviso: las ventanas se solapan 75 %, así que esta validación cruzada es
    # optimista. Sirve como cota superior, no como estimación honesta.
    sc = cross_val_score(RandomForestClassifier(200, random_state=0), Xs, yb, cv=5)
    print(f"    accuracy 5-fold: {sc.mean():.3f} ± {sc.std():.3f}   (azar = 0.500)")
    print("    OJO: ventanas solapadas al 75 %, esto es una COTA SUPERIOR optimista.")

    # ---- Veredicto -------------------------------------------------------
    print("\n" + "=" * 70)
    if cv_ax < 0.35 and abs(d_ax) < 0.3:
        print("  CAUSA MÁS PROBABLE: [A] el canal axial no lleva señal.")
        print("  log(tilt/axial) se comporta como proxy de amplitud de tilt, no")
        print("  como cociente entre dos movimientos. Nunca pudo discriminar.")
    if sc.mean() > 0.75 and utiles >= 3:
        print("\n  PERO las clases SÍ son separables con otras características.")
        print("  El problema es la característica elegida, no la tarea.")
    elif sc.mean() < 0.65:
        print("\n  Y las clases NO son separables ni con todas juntas: apunta a")
        print("  [B] o [C]. Antes de replantear el modelo hay que confirmar que")
        print("  el sujeto puede producir las dos técnicas de forma distinguible.")
    print("=" * 70)


if __name__ == "__main__":
    main()
