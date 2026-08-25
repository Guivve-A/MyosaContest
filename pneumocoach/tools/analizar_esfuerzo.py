"""¿Puede el esfuerzo disfrazarse de técnica?

    python tools/analizar_esfuerzo.py data/raw/s01_esfuerzo_<stamp>.csv

El primer sujeto real dejó un discriminador basado en amplitud absoluta. Eso
abre una pregunta que puede invalidar el enfoque entero: si una respiración
torácica floja produce la misma amplitud que una diafragmática exagerada,
entonces la amplitud mide intensidad y no técnica, y normalizarla no la salva.

El protocolo es un diseño factorial 2x2:

                    suave        fuerte
    diafragmática   dia_suave    dia_fuerte
    torácica        tor_suave    tor_fuerte

Con eso se separan dos efectos que en el protocolo largo estaban confundidos:

    efecto TÉCNICA    (dia vs tor, promediando esfuerzo)   <- lo que queremos medir
    efecto ESFUERZO   (suave vs fuerte, promediando técnica) <- el impostor

Una característica sirve si el efecto técnica supera al de esfuerzo. Si ocurre
lo contrario, esa característica está midiendo cuánto se esfuerza el paciente,
no si respira bien.
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

CELDAS = ["dia_suave", "dia_fuerte", "tor_suave", "tor_fuerte"]


def d(a, b):
    return (a.mean() - b.mean()) / (np.sqrt(0.5 * (a.var() + b.var())) + 1e-12)


def main():
    csv = Path(sys.argv[1])
    counts, et = cargar(csv)
    R, sesgo = cargar_calibracion(csv.stem.split("_")[0])
    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)

    X, y = [], []
    for tec, a, b in bloques(et):
        if b - a < C.WINDOW_N:
            continue
        for i in range(a, b - C.WINDOW_N + 1, C.HOP_N):
            X.append(dsp.extract_features({k: v[i:i + C.WINDOW_N] for k, v in ch.items()}))
            y.append(tec)
    X, y = np.stack(X), np.asarray(y)

    G = {c: X[y == c] for c in CELDAS if (y == c).sum() >= 3}
    faltan = [c for c in CELDAS if c not in G]
    if faltan:
        sys.exit(f"Faltan ventanas en: {faltan}. ¿Se grabó el protocolo --esfuerzo?")

    print("=" * 74)
    print("  DISEÑO FACTORIAL 2x2 · ventanas por celda")
    print("=" * 74)
    for c in CELDAS:
        print(f"    {c:<12} {len(G[c]):>3}")

    # ---- La comparación que decide -------------------------------------
    j_ti = C.FEATURE_NAMES.index("tilt_rms")
    j_ax = C.FEATURE_NAMES.index("axial_rms")

    print("\n" + "=" * 74)
    print("  LA PREGUNTA: ¿se confunde torácica floja con diafragmática fuerte?")
    print("=" * 74)
    print(f"\n  {'':16}{'tilt_rms (deg)':>18}{'axial_rms (mg)':>18}")
    for c in CELDAS:
        print(f"  {c:<16}{G[c][:, j_ti].mean():>18.3f}{G[c][:, j_ax].mean() * 1000:>18.3f}")

    print(f"\n  {'comparación':<34}{'d tilt':>10}{'d axial':>10}")
    pares = [
        ("tor_suave vs dia_fuerte  (IMPOSTOR)", "tor_suave", "dia_fuerte"),
        ("tor_fuerte vs dia_fuerte (técnica)", "tor_fuerte", "dia_fuerte"),
        ("tor_suave vs dia_suave   (técnica)", "tor_suave", "dia_suave"),
        ("dia_fuerte vs dia_suave  (esfuerzo)", "dia_fuerte", "dia_suave"),
        ("tor_fuerte vs tor_suave  (esfuerzo)", "tor_fuerte", "tor_suave"),
    ]
    for nombre, a, b in pares:
        print(f"  {nombre:<34}{d(G[a][:, j_ti], G[b][:, j_ti]):>+10.2f}"
              f"{d(G[a][:, j_ax], G[b][:, j_ax]):>+10.2f}")

    imp_ti = abs(d(G["tor_suave"][:, j_ti], G["dia_fuerte"][:, j_ti]))
    tec_ti = abs(d(G["tor_fuerte"][:, j_ti], G["dia_fuerte"][:, j_ti]))

    # ---- Técnica contra esfuerzo, característica a característica -------
    print("\n" + "=" * 74)
    print("  EFECTO TÉCNICA vs EFECTO ESFUERZO, por característica")
    print("=" * 74)
    print("""
  técnica  = |d| entre diafragmática y torácica, promediando esfuerzo
  esfuerzo = |d| entre suave y fuerte, promediando técnica
  Una característica sirve si técnica > esfuerzo.
""")
    dia = np.vstack([G["dia_suave"], G["dia_fuerte"]])
    tor = np.vstack([G["tor_suave"], G["tor_fuerte"]])
    sua = np.vstack([G["dia_suave"], G["tor_suave"]])
    fue = np.vstack([G["dia_fuerte"], G["tor_fuerte"]])

    filas = []
    for k, n in enumerate(C.FEATURE_NAMES):
        t = abs(d(dia[:, k], tor[:, k]))
        e = abs(d(sua[:, k], fue[:, k]))
        filas.append((t - e, t, e, n))
    filas.sort(reverse=True)

    print(f"  {'característica':<26}{'técnica':>9}{'esfuerzo':>10}{'margen':>9}")
    for margen, t, e, n in filas[:12]:
        marca = "  ROBUSTA" if (margen > 0.4 and t > 0.8) else ("  frágil" if margen < -0.4 else "")
        print(f"  {n:<26}{t:>9.2f}{e:>10.2f}{margen:>+9.2f}{marca}")
    print("  ...")
    for margen, t, e, n in filas[-4:]:
        print(f"  {n:<26}{t:>9.2f}{e:>10.2f}{margen:>+9.2f}   <- domina el esfuerzo")

    robustas = [n for m, t, e, n in filas if m > 0.4 and t > 0.8]

    # ---- Clasificador ciego al esfuerzo ---------------------------------
    print("\n" + "=" * 74)
    print("  ¿Puede un clasificador distinguir técnica IGNORANDO el esfuerzo?")
    print("=" * 74)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    Xb = np.vstack([dia, tor])
    yb = np.r_[np.zeros(len(dia)), np.ones(len(tor))]
    esf = np.r_[np.zeros(len(G["dia_suave"])), np.ones(len(G["dia_fuerte"])),
                np.zeros(len(G["tor_suave"])), np.ones(len(G["tor_fuerte"]))]

    # Entrenar SOLO con un nivel de esfuerzo y probar con el otro. Es la prueba
    # directa de si el modelo aprendió técnica o aprendió intensidad.
    Xs = StandardScaler().fit_transform(Xb)
    for tr_lvl, nombre in ((0, "entrena SUAVE  -> prueba FUERTE"),
                           (1, "entrena FUERTE -> prueba SUAVE")):
        tr, te = esf == tr_lvl, esf != tr_lvl
        m = RandomForestClassifier(300, random_state=0).fit(Xs[tr], yb[tr])
        acc = m.score(Xs[te], yb[te])
        print(f"  {nombre:<34}{acc:>8.3f}   (azar 0.500)")

    # Solo con las robustas
    if robustas:
        idx = [C.FEATURE_NAMES.index(n) for n in robustas]
        Xr = StandardScaler().fit_transform(Xb[:, idx])
        accs = []
        for tr_lvl in (0, 1):
            tr, te = esf == tr_lvl, esf != tr_lvl
            m = RandomForestClassifier(300, random_state=0).fit(Xr[tr], yb[tr])
            accs.append(m.score(Xr[te], yb[te]))
        print(f"  {'solo las robustas, promedio':<34}{np.mean(accs):>8.3f}"
              f"   ({len(robustas)} caracteristicas)")

    # ---- Veredicto -------------------------------------------------------
    print("\n" + "=" * 74)
    if imp_ti < 0.8:
        print("  >> LA AMPLITUD SE DEJA ENGAÑAR.")
        print(f"     Torácica floja y diafragmática fuerte se separan solo por")
        print(f"     d = {imp_ti:.2f} en tilt_rms, contra d = {tec_ti:.2f} de la")
        print("     comparación honesta a igual esfuerzo. Un paciente que se")
        print("     esfuerce poco puede ser clasificado como correcto haciéndolo")
        print("     mal, y al revés.")
    else:
        print("  >> LA AMPLITUD AGUANTA.")
        print(f"     Incluso a esfuerzos opuestos, torácica y diafragmática se")
        print(f"     separan con d = {imp_ti:.2f}. La intensidad modula pero no")
        print("     borra la firma de la técnica.")
    if robustas:
        print(f"\n     {len(robustas)} características tienen más efecto de técnica")
        print(f"     que de esfuerzo: {', '.join(robustas[:6])}")
    else:
        print("\n     NINGUNA característica supera al efecto del esfuerzo.")
    print("=" * 74)


if __name__ == "__main__":
    main()
