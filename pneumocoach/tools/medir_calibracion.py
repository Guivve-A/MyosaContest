"""Mide cuánto gana la calibración por sesión, con las grabaciones reales.

    python tools/medir_calibracion.py

La evaluación es leave-one-session-out: se entrena con dos sesiones y se prueba
en la tercera, que es la generalización más honesta que permiten tres sesiones
del mismo sujeto.

Reglas que hacen honesta la medición:

  * Las ventanas de referencia salen SOLO de los primeros REF_SEGUNDOS de cada
    maniobra, simulando lo que el dispositivo recogería en la calibración real.
  * Esas ventanas se excluyen del conjunto de prueba. Evaluar sobre los datos
    que definieron el eje sería medirse a sí mismo.
  * La referencia de la sesión de prueba se calcula con SUS PROPIOS datos de
    calibración, nunca con los de entrenamiento: es lo que pasaría en la clínica.
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
from scipy import stats  # noqa: E402
from pneumocoach.calibracion import REF_SEGUNDOS, ReferenciaSesion  # noqa: E402
from analizar_captura import bloques, cargar, cargar_calibracion  # noqa: E402

# Grupo de protocolo de cada sesion. Dos sesiones del mismo grupo comparten
# consignas y estructura, asi que se predicen entre si por parecido y no por
# generalizacion: entrenar con una y probar en la otra mide similitud.
#
# Dejar de sacarlas juntas del entrenamiento fue lo que inflo el pliegue s3 en
# agosto, y volveria a inflar s1 y s4 ahora que las dos usan el protocolo largo.
# Por eso la validacion es dejar-un-GRUPO-fuera, no dejar-una-sesion-fuera.
GRUPO = {
    "s1 protocolo": "protocolo",
    "s2 esfuerzo": "esfuerzo",
    "s3 esfuerzo": "esfuerzo",
    "s4 protocolo": "protocolo",
}

SESIONES = {
    "s1 protocolo": "data/raw/s01_protocolo_20260818_174857.csv",
    "s2 esfuerzo": "data/raw/s01_esfuerzo_20260818_181830.csv",
    "s3 esfuerzo": "data/raw/s01_esfuerzo_20260818_183515.csv",
    # 22 ago: consignas de dos manos y montaje remedido (perpendicular 1.8
    # frente a 13.8 grados). Mismo protocolo que s1, asi que s4 y s1 son el
    # par comparable y los pliegues cruzados entre ambos son los defendibles.
    "s4 protocolo": "data/raw/s01_protocolo_20260822_120903.csv",
}

ES_DIA = ("diaphragmatic", "dia_suave", "dia_fuerte")
ES_TOR = ("thoracic", "tor_suave", "tor_fuerte")


def cargar_sesion(path: str, R=None, sesgo=None):
    """Devuelve (X, y_binaria, es_referencia) por ventana.

    `es_referencia` marca las ventanas que caen en los primeros REF_SEGUNDOS de
    cada bloque: son las que el dispositivo usaría para calibrar.
    """
    counts, et = cargar(REPO / path)
    # Cada grabacion con la matriz de montaje que le toca por fecha.
    import json as _json
    _meta = _json.loads((REPO / path).with_suffix('.json').read_text(encoding='utf-8'))
    R, sesgo = cargar_calibracion('s01', _meta.get('inicio_utc'))
    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)
    X, y, ref = [], [], []
    ref_n = int(REF_SEGUNDOS * C.FS_HZ)

    for tec, a, b in bloques(et):
        if tec not in ES_DIA + ES_TOR or b - a < C.WINDOW_N:
            continue
        for i in range(a, b - C.WINDOW_N + 1, C.HOP_N):
            X.append(dsp.extract_features({k: v[i:i + C.WINDOW_N] for k, v in ch.items()}))
            y.append(0 if tec in ES_DIA else 1)
            # La ventana es de referencia si TERMINA dentro de los primeros
            # REF_SEGUNDOS del bloque.
            ref.append((i + C.WINDOW_N) - a <= ref_n + C.WINDOW_N)
    return np.stack(X), np.asarray(y), np.asarray(ref)


def evaluar(usar_calibracion: bool, D, solo=None) -> tuple[float, list[float]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    accs, crudo = [], []
    grupos = sorted({GRUPO[k] for k in D})
    for g in grupos:
        te_keys = [k for k in D if GRUPO[k] == g]
        tr_keys = [k for k in D if GRUPO[k] != g]
        if not tr_keys:
            continue

        def prep(k):
            X, y, ref = D[k]
            if not usar_calibracion:
                return X, y, ref
            r = ReferenciaSesion.desde_ventanas(X[ref & (y == 0)], X[ref & (y == 1)])
            return r.normaliza(X), y, ref

        Xtr = np.vstack([prep(k)[0][~D[k][2]] for k in tr_keys])
        ytr = np.concatenate([D[k][1][~D[k][2]] for k in tr_keys])
        # Cada sesion de prueba se normaliza con SU PROPIA referencia, que es
        # lo que ocurriria en la clinica.
        Xte = np.vstack([prep(k)[0][~D[k][2]] for k in te_keys])
        yte = np.concatenate([D[k][1][~D[k][2]] for k in te_keys])

        if solo is not None:
            Xtr, Xte = Xtr[:, solo], Xte[:, solo]

        sc = StandardScaler().fit(Xtr)
        m = RandomForestClassifier(300, random_state=0, min_samples_leaf=3)
        m.fit(sc.transform(Xtr), ytr)
        pred = m.predict(sc.transform(Xte))
        accs.append(float((pred == yte).mean()))
        crudo.append((int((pred == yte).sum()), int(len(yte))))
    return float(np.mean(accs)), accs, grupos, crudo


def main():
    # La matriz se elige por sesion dentro de cargar_sesion().
    R = sesgo = None
    D = {k: cargar_sesion(v, R, sesgo) for k, v in SESIONES.items()}

    print("=" * 76)
    print("  CALIBRACIÓN POR SESIÓN · leave-one-session-out")
    print("=" * 76)
    print(f"\n  Referencia: primeros {REF_SEGUNDOS:.0f} s de cada maniobra, "
          f"excluidos de la prueba.\n")
    print(f"  {'sesión':<16}{'ventanas':>10}{'de referencia':>15}{'de prueba':>12}")
    for k, (X, y, ref) in D.items():
        print(f"  {k:<16}{len(X):>10}{int(ref.sum()):>15}{int((~ref).sum()):>12}")

    # Calidad de cada calibración
    print(f"\n  {'sesión':<16}{'caract. informativas':>22}{'contraste mediano':>20}")
    for k, (X, y, ref) in D.items():
        r = ReferenciaSesion.desde_ventanas(X[ref & (y == 0)], X[ref & (y == 1)])
        q = r.calidad()
        print(f"  {k:<16}{q['caracteristicas_informativas']:>17} / {C.N_FEATURES}"
              f"{q['contraste_mediano']:>20.3f}")

    sin, sin_f, grupos, _ = evaluar(False, D)
    con, con_f, _, crudo = evaluar(True, D)

    print("\n" + "=" * 76)
    print("  RESULTADO  ·  dejar-un-GRUPO-de-protocolo-fuera")
    print("=" * 76)
    print(f"\n{'prueba en':<22}{'sin calibrar':>16}"
          f"{'con calibración':>18}{'ganancia':>12}")
    for g, a, b in zip(grupos, sin_f, con_f):
        miembros = ", ".join(k.split()[0] for k in D if GRUPO[k] == g)
        print(f"  {g + ' (' + miembros + ')':<22}{a:>16.3f}{b:>18.3f}{b - a:>+12.3f}")
    print(f"  {'PROMEDIO':<22}{sin:>16.3f}{con:>18.3f}{con - sin:>+12.3f}")
    print(f"\nazar = 0.500")

    j = [C.FEATURE_NAMES.index("axial_rms")]
    sin1, _, _, _ = evaluar(False, D, solo=j)
    con1, _, _, _ = evaluar(True, D, solo=j)
    print(f"\nSolo axial_rms:   sin calibrar {sin1:.3f}   con calibración {con1:.3f}")

    print("\n" + "=" * 76)
    # El titular es el agregado ponderado por ventanas, no la media simple de
    # los pliegues: uno tiene 48 ventanas y el otro 208, y promediarlos a partes
    # iguales le da al pequeno cuatro veces mas peso del que le corresponde.
    ac = sum(c[0] for c in crudo)
    nv = sum(c[1] for c in crudo)
    p_hat = ac / nv
    # Ventanas solapadas al 75 %: cuatro consecutivas comparten datos, asi que
    # el n independiente es la cuarta parte. Sin esta correccion el intervalo
    # sale la mitad de ancho de lo que es.
    n_eff = nv / (C.WINDOW_N / C.HOP_N)
    lo, hi = stats.beta.ppf([0.025, 0.975],
                            p_hat * n_eff + 0.5, (1 - p_hat) * n_eff + 0.5)
    pv = stats.binomtest(round(p_hat * n_eff), round(n_eff), 0.5,
                         alternative="greater").pvalue
    print(f"  CIFRA DEFENDIBLE: {p_hat:.3f}   ({ac}/{nv} ventanas)")
    print(f"  IC 95 % [{lo:.3f}, {hi:.3f}]  ·  n efectivo {n_eff:.0f}  ·  p = {pv:.4f}")
    print(f"  {'SIGNIFICATIVO frente al azar de 0.500' if lo > 0.5 else 'NO SIGNIFICATIVO: el intervalo incluye el azar'}")
    print()
    print("  Validacion dejar-un-GRUPO-de-protocolo-fuera: ningun pliegue")
    print("  se entrena con una sesion de su mismo protocolo. Eso era lo que")
    print("  inflaba las cifras anteriores, primero con s3 y luego con s1+s4.")
    print()
    for g, a, b in zip(grupos, sin_f, con_f):
        print(f"    probando en {g:<12} {b:.3f}  (sin calibrar {a:.3f})")

    print()
    if con - sin > 0.08:
        print(f"  La calibracion por sesion aporta {(con - sin) * 100:+.1f} puntos.")
    elif con > sin:
        print(f"  La calibracion aporta solo {(con - sin) * 100:+.1f} puntos. Marginal.")
    else:
        print(f"  La calibracion NO ayuda ({(con - sin) * 100:+.1f} puntos).")
    print("""
  Recordatorio de alcance: son cuatro sesiones de UN sujeto, y con 75 % de
  solape el n efectivo ronda un cuarto del nominal. Esto mide estabilidad
  entre sesiones y entre protocolos, no generalizacion entre personas, que
  sigue sin medirse.""")
    print("=" * 76)


if __name__ == "__main__":
    main()
