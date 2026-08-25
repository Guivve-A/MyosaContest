"""¿Cuánto puede bajar la latencia del veredicto sin perder exactitud?

    python tools/medir_latencia.py

La pregunta detrás
------------------
«Hacer la inferencia más rápida» suena a optimizar el modelo, y ahí no hay nada
que ganar: TFLM tarda 224 µs y el DSP 1210 µs, sobre un salto de 3000 ms. El
cómputo es el 0.05 % del presupuesto.

Lo que de verdad marca cuánto tarda el dispositivo en reaccionar son dos
parámetros del contrato:

    VENTANA  cuánta señal resume cada veredicto
    SALTO    cada cuánto sale un veredicto nuevo

Un cambio de técnica no se refleja del todo hasta que la ventana se ha llenado
con la técnica nueva, y luego hay que esperar al siguiente veredicto. Es decir:

    latencia ≈ ventana + salto

Con 12 s y 3 s son 15 s. Bajar eso es posible, pero no es gratis y aquí se mide
qué cuesta.

El suelo físico
---------------
A las 6-14 respiraciones por minuto que persigue el coaching, una ventana de
12 s contiene UNA O DOS respiraciones. Acortarla deja ventanas con menos de una
respiración completa, y las cuatro características de temporización dejan de
significar nada. Ese límite no lo pone el procesador: lo pone el paciente.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401
from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402
from pneumocoach.calibracion import ReferenciaSesion  # noqa: E402
from analizar_captura import bloques, cargar, cargar_calibracion  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import json  # noqa: E402
from scipy import stats  # noqa: E402

SESIONES = {
    "s1": ("data/raw/s01_protocolo_20260818_174857.csv", "protocolo"),
    "s2": ("data/raw/s01_esfuerzo_20260818_181830.csv", "esfuerzo"),
    "s3": ("data/raw/s01_esfuerzo_20260818_183515.csv", "esfuerzo"),
    "s4": ("data/raw/s01_protocolo_20260822_120903.csv", "protocolo"),
}
A_CLASE = {
    "diaphragmatic": 0, "dia_suave": 0, "dia_fuerte": 0,
    "thoracic": 1, "tor_suave": 1, "tor_fuerte": 1,
    "rapid_shallow": 2,
}
ES_DIA = ("diaphragmatic", "dia_suave", "dia_fuerte")
ES_TOR = ("thoracic", "tor_suave", "tor_fuerte")

# ventana_s, salto_s
CONFIGS = [
    (12.0, 3.0),    # el actual
    (12.0, 1.5),
    (12.0, 0.75),
    (10.0, 2.0),
    (8.0, 2.0),
    (6.0, 1.5),
]


def aplica(ventana_s: float, salto_s: float) -> None:
    """Reescribe el contrato en memoria y recarga el DSP.

    La ventana de Hann y la rejilla de frecuencias se calculan al importar el
    módulo, así que sin recargar se seguirían usando las del tamaño anterior y
    el experimento mediría otra cosa.
    """
    C.WINDOW_S = ventana_s
    C.WINDOW_N = int(ventana_s * C.FS_HZ)
    C.HOP_S = salto_s
    C.HOP_N = int(salto_s * C.FS_HZ)
    C.DEC_N = C.WINDOW_N // C.DECIM
    importlib.reload(dsp)


def carga(path: str) -> dict:
    p = REPO / path
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    R, sesgo = cargar_calibracion("s01", meta.get("inicio_utc"))
    counts, et = cargar(p)
    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)

    X, y, ref, resp = [], [], [], []
    ref_n = int(C.REF_SEGUNDOS * C.FS_HZ)
    for tec, a, b in bloques(et):
        if tec not in A_CLASE:
            continue
        for i in range(a, b - C.WINDOW_N + 1, C.HOP_N):
            w = {k: v[i:i + C.WINDOW_N] for k, v in ch.items()}
            X.append(dsp.extract_features(w))
            y.append(A_CLASE[tec])
            ref.append(tec in ES_DIA + ES_TOR and (i + C.WINDOW_N) - a <= ref_n)
            per, _ = dsp.segment_breaths(w["tilt"], C.FS_HZ)
            resp.append(len(per))
    X = np.stack(X)
    y = np.asarray(y)
    ref = np.asarray(ref)
    r = ReferenciaSesion.desde_ventanas(X[ref & (y == 0)], X[ref & (y == 1)])
    return {"Z": r.normaliza(X).astype(np.float32), "y": y, "ref": ref,
            "resp": np.asarray(resp)}


def evalua(ses: dict) -> tuple[float, int]:
    """Binaria diafragmática/torácica, dejando un grupo de protocolo fuera."""
    ac = nv = 0
    for g in sorted({v[1] for v in SESIONES.values()}):
        te = [k for k in ses if SESIONES[k][1] == g]
        tr = [k for k in ses if SESIONES[k][1] != g]
        Ztr = np.vstack([ses[k]["Z"][~ses[k]["ref"]] for k in tr])
        ytr = np.concatenate([ses[k]["y"][~ses[k]["ref"]] for k in tr])
        Zte = np.vstack([ses[k]["Z"][~ses[k]["ref"]] for k in te])
        yte = np.concatenate([ses[k]["y"][~ses[k]["ref"]] for k in te])
        sc = StandardScaler().fit(Ztr)
        m = MLPClassifier(C.MLP_HIDDEN, max_iter=2000, alpha=0.1, random_state=0)
        m.fit(sc.transform(Ztr), ytr)
        prob = m.predict_proba(sc.transform(Zte))
        cls = list(m.classes_)
        b = np.isin(yte, [0, 1])
        if 0 not in cls or 1 not in cls:
            continue
        p = np.where(prob[b][:, cls.index(0)] >= prob[b][:, cls.index(1)], 0, 1)
        ac += int((p == yte[b]).sum())
        nv += int(b.sum())
    return (ac / nv if nv else float("nan")), nv


def main() -> int:
    v0, s0 = C.WINDOW_S, C.HOP_S
    print("=" * 78)
    print("  LATENCIA CONTRA EXACTITUD")
    print("=" * 78)
    print()
    print(f"  {'ventana':>8}{'salto':>7}{'latencia':>9}{'binaria':>9}"
          f"{'IC 95 %':>18}{'n ef':>6}{'resp':>6}{'<1 resp':>9}")
    print("  " + "-" * 76)

    filas = []
    for ventana, salto in CONFIGS:
        aplica(ventana, salto)
        ses = {k: carga(p) for k, (p, _) in SESIONES.items()}
        acc, n = evalua(ses)
        resp = np.concatenate([ses[k]["resp"] for k in ses])
        pocas = float((resp < 1).mean())
        lat = ventana + salto
        # n efectivo: las ventanas se solapan, asi que el numero independiente
        # es n dividido por cuantas comparten datos. Sin esta correccion un
        # salto pequeno parece dar mucha mas evidencia de la que da.
        n_ef = n / (ventana / salto)
        lo, hi = stats.beta.ppf([0.025, 0.975], acc * n_ef + 0.5,
                                (1 - acc) * n_ef + 0.5)
        marca = " <- actual" if (ventana, salto) == (v0, s0) else ""
        print(f"  {ventana:>7.0f}s{salto:>6.2f}s{lat:>8.1f}s{acc:>9.3f}"
              f"   [{lo:.3f}, {hi:.3f}]{n_ef:>6.0f}{np.median(resp):>6.1f}"
              f"{pocas:>8.0%}{marca}")
        filas.append((ventana, salto, lat, acc, pocas, lo, hi))

    aplica(v0, s0)
    base = next(f for f in filas if (f[0], f[1]) == (v0, s0))
    print("  " + "-" * 76)
    print()
    print(f"  Referencia: ventana {base[0]:.0f} s, salto {base[1]:.2f} s, "
          f"latencia {base[2]:.1f} s, binaria {base[3]:.3f}")
    print()

    # Una configuracion es DESCARTABLE solo si su intervalo no se solapa con el
    # del actual. Comparar dos puntos sin mirar sus intervalos es como se
    # eligen parametros por ruido.
    print("  Configuraciones cuya exactitud NO se distingue de la actual")
    print("  (intervalos solapados), ordenadas por latencia:")
    print()
    solapan = [f for f in filas if f[6] >= base[5] and f[5] <= base[6]]
    for f in sorted(solapan, key=lambda f: f[2]):
        aviso = ("  ATENCION: " + f"{f[4]:.0%} de ventanas con menos de una "
                 "respiracion") if f[4] > 0.10 else ""
        print(f"    ventana {f[0]:>4.0f} s  salto {f[1]:>5.2f} s  "
              f"latencia {f[2]:>5.1f} s{aviso}")
    print()
    peores = [f for f in filas if f[6] < base[5]]
    if peores:
        print("  Medibles como PEORES (su intervalo queda entero por debajo):")
        for f in peores:
            print(f"    ventana {f[0]:>4.0f} s  salto {f[1]:>5.2f} s  "
                  f"binaria {f[3]:.3f}  IC hasta {f[6]:.3f}")
    print()
    print("  El coste del computo no aparece en esta tabla porque no importa:")
    print("  1.4 ms por ventana contra un salto de cientos o miles. Aunque el")
    print("  modelo fuera instantaneo, la latencia bajaria un 0.05 %.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
