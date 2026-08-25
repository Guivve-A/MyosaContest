"""Entrena el modelo sobre las grabaciones REALES y lo exporta a INT8.

    python ml/scripts/entrenar_real.py

Sustituye a scripts/train.py, que entrenaba sobre el generador sintético. Ese
generador diverge 15-16 sigma de los datos reales y su modelo está retirado en
ml/artifacts/RETIRADO/.

Cómo se entrena y por qué así
------------------------------
El modelo NO ve características absolutas: ve la proyección sobre el eje del
propio paciente, `z = (x - ref_dia) / (ref_tor - ref_dia)`. El dispositivo
calcula ese mismo `z` a bordo, así que el orden de la cadena es idéntico en los
dos lados:

    29 características -> z (calibración de sesión) -> estandarizado -> INT8

Las ventanas de referencia -las que definieron el eje- se excluyen de
entrenamiento y de prueba. Evaluar sobre ellas sería medirse a sí mismo.

La partición es por GRUPO DE PROTOCOLO, no por sesión. Dos sesiones del mismo
protocolo comparten consignas y se predicen entre sí por parecido; dejar una
fuera y entrenar con su gemela mide similitud, no generalización. Es el error
que infló las cifras anteriores dos veces.

Qué se puede afirmar de cada clase
-----------------------------------
`rapid_shallow` y `artifact` SOLO aparecen en el protocolo largo. El pliegue que
entrena con esfuerzo no las ve nunca, así que no tienen validación cruzada entre
protocolos y el informe lo dice. La cifra defendible del sistema es la binaria
diafragmática/torácica, que sí está en los dos grupos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401
from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402
from pneumocoach.calibracion import ReferenciaSesion  # noqa: E402
from pneumocoach.train import (  # noqa: E402
    Standardiser, evaluate, print_confusion, quantise_int8, tflite_predict,
    tflite_probabilities, train_mlp,
)
from pneumocoach.dataset import Dataset  # noqa: E402
from analizar_captura import bloques, cargar, cargar_calibracion  # noqa: E402
from scipy import stats  # noqa: E402

SESIONES = {
    "s1 protocolo": ("data/raw/s01_protocolo_20260818_174857.csv", "protocolo"),
    "s2 esfuerzo":  ("data/raw/s01_esfuerzo_20260818_181830.csv",  "esfuerzo"),
    "s3 esfuerzo":  ("data/raw/s01_esfuerzo_20260818_183515.csv",  "esfuerzo"),
    "s4 protocolo": ("data/raw/s01_protocolo_20260822_120903.csv", "protocolo"),
}

# Las etiquetas de intensidad son la misma técnica ejecutada con más o menos
# esfuerzo. La técnica es lo que se clasifica; la intensidad es la variable que
# el protocolo de esfuerzo introduce a propósito para comprobar que no la
# falsifica.
A_CLASE = {
    "diaphragmatic": "diaphragmatic", "dia_suave": "diaphragmatic",
    "dia_fuerte": "diaphragmatic",
    "thoracic": "thoracic", "tor_suave": "thoracic", "tor_fuerte": "thoracic",
    "rapid_shallow": "rapid_shallow",
}
# Reposo y apnea quedan fuera del entrenamiento a propósito: no son técnicas
# que se coacheen, y medimos que el reposo NO es separable de una maniobra
# deliberada (0.460 cruzando protocolos, por debajo del azar). Se conservan para
# medir qué hace el modelo cuando las ve, en vez de suponerlo.
FUERA = ("reposo", "apnea", "artifact")

ES_DIA = ("diaphragmatic", "dia_suave", "dia_fuerte")
ES_TOR = ("thoracic", "tor_suave", "tor_fuerte")


def carga_sesion(path: str) -> dict:
    """Ventanas de una sesión, ya proyectadas sobre el eje del paciente."""
    p = REPO / path
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    R, sesgo = cargar_calibracion("s01", meta.get("inicio_utc"))
    counts, et = cargar(p)
    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)

    X, etiquetas, es_ref = [], [], []
    ref_n = int(C.REF_SEGUNDOS * C.FS_HZ)
    for tec, a, b in bloques(et):
        if tec not in A_CLASE and tec not in FUERA:
            continue
        for i in range(a, b - C.WINDOW_N + 1, C.HOP_N):
            X.append(dsp.extract_features(
                {k: v[i:i + C.WINDOW_N] for k, v in ch.items()}))
            etiquetas.append(tec)
            # Referencia si la ventana CABE ENTERA en los primeros REF_SEGUNDOS
            # del bloque, que es lo que recogería el dispositivo.
            es_ref.append(tec in ES_DIA + ES_TOR
                          and (i + C.WINDOW_N) - a <= ref_n)
    X = np.stack(X)
    etiquetas = np.asarray(etiquetas)
    es_ref = np.asarray(es_ref)

    ref = ReferenciaSesion.desde_ventanas(
        X[es_ref & np.isin(etiquetas, ES_DIA)],
        X[es_ref & np.isin(etiquetas, ES_TOR)],
    )
    return {"Z": ref.normaliza(X).astype(np.float32), "etiquetas": etiquetas,
            "es_ref": es_ref, "calidad": ref.calidad()}


def a_dataset(sesiones: dict, claves) -> tuple[np.ndarray, np.ndarray]:
    """Matriz y etiquetas de clase, excluyendo referencia y clases fuera."""
    Z, y = [], []
    for k in claves:
        d = sesiones[k]
        util = ~d["es_ref"] & np.isin(d["etiquetas"], list(A_CLASE))
        Z.append(d["Z"][util])
        y.append([C.CLASS_INDEX[A_CLASE[t]] for t in d["etiquetas"][util]])
    return np.vstack(Z), np.concatenate(y).astype(np.int64)


def ic_binomial(p_hat: float, n_ef: float) -> tuple[float, float, float]:
    """Intervalo de Jeffreys sobre el n EFECTIVO, no el nominal.

    Toma la PROPORCION y el n efectivo, y de ahi saca el numero de aciertos
    equivalente. La primera version recibia la proporcion y la usaba como si
    fuera un recuento, lo que daba intervalos imposibles -[0.000, 0.046] para
    una exactitud de 0.525- que se ven a simple vista si uno mira el numero en
    vez de confiar en la formula.
    """
    exitos = p_hat * n_ef
    lo, hi = stats.beta.ppf([0.025, 0.975], exitos + 0.5, n_ef - exitos + 0.5)
    pv = stats.binomtest(round(exitos), round(n_ef), 0.5,
                         alternative="greater").pvalue
    return lo, hi, pv


def valida_por_grupo(sesiones: dict, seed: int) -> dict:
    """Deja un grupo de protocolo fuera y mide. Es la cifra defendible."""
    grupos = sorted({g for _, g in SESIONES.values()})
    total_bin = [0, 0]
    por_grupo = {}
    for g in grupos:
        te = [k for k in sesiones if SESIONES[k][1] == g]
        tr = [k for k in sesiones if SESIONES[k][1] != g]
        Ztr, ytr = a_dataset(sesiones, tr)
        Zte, yte = a_dataset(sesiones, te)
        std = Standardiser.fit(Ztr)
        modelo = entrena(std(Ztr), ytr, seed)
        prob = modelo.predict(std(Zte), verbose=0)
        pred = np.argmax(prob, axis=1)

        # Dos preguntas distintas, y hay que medirlas por separado.
        #
        # BINARIA: dada una ventana que ES diafragmatica o toracica, .acierta el
        # modelo cual de las dos? Se decide entre esas dos salidas y nada mas.
        # Es la pregunta clinica central y la unica comparable con las cifras
        # anteriores del proyecto, que se midieron sobre un modelo binario.
        #
        # COMPLETA: lo que el dispositivo hace de verdad, eligiendo entre las
        # cinco clases. Es mas baja por construccion, y en el pliegue que
        # entrena con esfuerzo lo es aun mas porque ese grupo no contiene ni
        # rapid_shallow ni artifact: el modelo no puede acertar lo que no vio.
        i_dia, i_tor = C.CLASS_INDEX["diaphragmatic"], C.CLASS_INDEX["thoracic"]
        bin_te = np.isin(yte, [i_dia, i_tor])
        pred_bin = np.where(prob[bin_te][:, i_dia] >= prob[bin_te][:, i_tor],
                            i_dia, i_tor)
        ac = int((pred_bin == yte[bin_te]).sum())
        nv = int(bin_te.sum())
        total_bin[0] += ac
        total_bin[1] += nv
        clases_tr = sorted({C.CLASSES[i].key for i in np.unique(ytr)})
        por_grupo[g] = {
            "binaria": ac / nv if nv else float("nan"),
            "n": nv,
            "clases_entrenadas": clases_tr,
            "completa": float((pred == yte).mean()),
            "fuera_de_binaria": float(
                (~np.isin(pred[bin_te], [i_dia, i_tor])).mean()),
        }
    ac, nv = total_bin
    n_ef = nv / (C.WINDOW_N / C.HOP_N)
    lo, hi, pv = ic_binomial(ac / nv, n_ef)
    return {"por_grupo": por_grupo, "aciertos": ac, "ventanas": nv,
            "acc": ac / nv, "n_efectivo": n_ef, "ic": (lo, hi), "p": pv}


def construye_mlp(seed: int):
    """Esqueleto Keras del MLP: solo Dense y Softmax.

    Es lo unico que registra el resolutor del firmware
    (MicroMutableOpResolver<2> con FULLY_CONNECTED y SOFTMAX). Cualquier capa
    extra compilaria aqui y fallaria en el dispositivo al buscar el operador.
    """
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    capas = [tf.keras.layers.Input(shape=(C.N_FEATURES,))]
    for i, u in enumerate(C.MLP_HIDDEN):
        capas.append(tf.keras.layers.Dense(u, activation="relu", name=f"dense_{i}"))
    capas.append(tf.keras.layers.Dense(C.N_CLASSES, activation="softmax",
                                       name="verdict"))
    return tf.keras.Sequential(capas, name="pneumocoach")


def entrena(Xtr, ytr, seed: int):
    """Entrena con scikit-learn y trasplanta los pesos a Keras.

    Puede parecer un rodeo, y es lo contrario: el modelo que se despliega es
    EXACTAMENTE el que se valido. Medido bajo la misma particion por grupo,
    MLPClassifier(32,16, alpha=0.1) da 0.750 binario cruzando protocolos; el
    mismo esqueleto entrenado en Keras se quedaba en 0.56-0.63 porque corria
    hasta converger y sobreajustaba 400 ventanas. La diferencia no es la
    arquitectura -es identica- sino el criterio de parada: sklearn se detiene
    cuando la perdida se estanca.

    Reproducir esa receta en Keras era posible, pero trasplantar los pesos deja
    cero distancia entre lo medido y lo desplegado, que es lo que importa.

    Keras solo se usa como formato de salida hacia el conversor TFLite: la
    arquitectura es Dense(32) -> Dense(16) -> Dense(3, softmax), que es lo mismo
    que scikit-learn entrena.
    """
    import numpy as np
    from sklearn.neural_network import MLPClassifier

    sk = MLPClassifier(C.MLP_HIDDEN, max_iter=2000, alpha=0.1,
                       random_state=seed)
    sk.fit(Xtr, ytr)

    modelo = construye_mlp(seed)
    capas = [c for c in modelo.layers if hasattr(c, "kernel")]
    ocultas, salida = capas[:-1], capas[-1]
    for capa, W, b in zip(ocultas, sk.coefs_, sk.intercepts_):
        capa.set_weights([W.astype("float32"), b.astype("float32")])

    # La capa de salida hay que reconstruirla, no copiarla.
    #
    # Con dos clases, sklearn no usa softmax: usa UNA salida logistica. Y aunque
    # use softmax, sus columnas corresponden a las clases que vio, que no tienen
    # por que ser las N del contrato. El dispositivo siempre espera N salidas en
    # el orden de C.CLASSES.
    #
    # softmax sobre [0, z] reproduce exactamente sigmoid(z), asi que el caso
    # binario se traduce sin perdida. Las clases ausentes reciben un logit muy
    # negativo: existen en el vector de salida y nunca ganan.
    W_sk, b_sk = sk.coefs_[-1], sk.intercepts_[-1]
    n_oculta = W_sk.shape[0]
    W = np.zeros((n_oculta, C.N_CLASSES), dtype="float32")
    b = np.full(C.N_CLASSES, -30.0, dtype="float32")
    clases = list(sk.classes_)
    if W_sk.shape[1] == 1:                      # binario, salida logistica
        c0, c1 = int(clases[0]), int(clases[1])
        W[:, c0] = 0.0;            b[c0] = 0.0
        W[:, c1] = W_sk[:, 0];     b[c1] = b_sk[0]
    else:                                       # multiclase, columna por clase
        for j, c in enumerate(clases):
            W[:, int(c)] = W_sk[:, j]
            b[int(c)] = b_sk[j]
    salida.set_weights([W, b])

    # Comprobacion obligatoria: si el trasplante no reproduce a sklearn, lo
    # desplegado no es lo validado y todo lo anterior deja de significar nada.
    p_sk_raw = sk.predict_proba(Xtr)
    p_sk = np.zeros((len(Xtr), C.N_CLASSES))
    for j, c in enumerate(clases):
        p_sk[:, int(c)] = p_sk_raw[:, j]
    p_ks = modelo.predict(Xtr, verbose=0)
    dif = float(np.abs(p_sk - p_ks).max())
    print(f"  trasplante de pesos: max |dif| de probabilidad {dif:.2e}"
          f"   ({len(clases)} clases en entrenamiento)")
    assert dif < 1e-4, "el modelo Keras no reproduce a sklearn"
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modelo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=REPO / "ml" / "artifacts")
    args = ap.parse_args()

    print("=" * 74)
    print("  ENTRENAMIENTO SOBRE DATOS REALES")
    print("=" * 74)

    sesiones = {}
    for k, (path, grupo) in SESIONES.items():
        sesiones[k] = carga_sesion(path)
        q = sesiones[k]["calidad"]
        print(f"  {k:<16}{len(sesiones[k]['Z']):>5} ventanas   "
              f"ref {int(sesiones[k]['es_ref'].sum()):>3}   "
              f"contraste {q['contraste_mediano']:.3f}")

    Z, y = a_dataset(sesiones, list(sesiones))
    print(f"\n  entrenamiento: {len(Z)} ventanas")
    for c in C.CLASSES:
        print(f"    {c.key:<16}{int((y == c.idx).sum()):>5}")

    print("\n" + "=" * 74)
    print("  VALIDACION  ·  dejar-un-GRUPO-de-protocolo-fuera")
    print("=" * 74)
    v = valida_por_grupo(sesiones, args.seed)
    for g, r in v["por_grupo"].items():
        falta = [c.key for c in C.CLASSES if c.key not in r["clases_entrenadas"]]
        nota = f"   sin {', '.join(falta)} en entrenamiento" if falta else ""
        print(f"  probando en {g:<12} binaria {r['binaria']:.3f} "
              f"({r['n']} ventanas){nota}")
    lo, hi = v["ic"]
    print(f"\n  BINARIA diafragmatica/toracica: {v['acc']:.3f}  "
          f"({v['aciertos']}/{v['ventanas']})")
    print(f"  IC 95 % [{lo:.3f}, {hi:.3f}]   n efectivo {v['n_efectivo']:.0f}   "
          f"p = {v['p']:.4f}")
    print(f"  {'SIGNIFICATIVA' if lo > 0.5 else 'NO significativa'} frente al azar 0.500")
    print()
    print("  rapid_shallow y artifact solo existen en el protocolo largo, asi que")
    print("  no tienen validacion cruzada entre protocolos. El modelo las aprende")
    print("  y el dispositivo las usara, pero su exactitud no esta medida fuera")
    print("  del protocolo donde se grabaron.")

    print("\n" + "=" * 74)
    print("  MODELO FINAL  ·  entrenado con las cuatro sesiones")
    print("=" * 74)
    std = Standardiser.fit(Z)
    modelo = entrena(std(Z), y, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    blob = quantise_int8(modelo, std(Z), args.out)
    (args.out / "model_int8.tflite").write_bytes(blob)
    np.savez(args.out / "standardiser.npz", mean=std.mean, scale=std.scale)

    pred = tflite_predict(blob, std(Z))
    m = evaluate(y, pred)
    print(f"\n  tamano del modelo: {len(blob)} bytes "
          f"(presupuesto {C.MAX_MODEL_BYTES})")
    print(f"  exactitud INT8 sobre su PROPIO entrenamiento: {m.accuracy:.3f}")
    print("  (no es una cifra de rendimiento; solo confirma que la cuantizacion")
    print("   no rompio el modelo. La cifra del sistema es la de arriba.)")
    print()
    print_confusion(m)

    # Que hace el modelo con lo que no se le enseno.
    print("\n" + "=" * 74)
    print("  REPOSO Y APNEA  ·  clases que el modelo NO tiene")
    print("=" * 74)
    Zf, nombres = [], []
    for k, d in sesiones.items():
        m_f = np.isin(d["etiquetas"], FUERA)
        if m_f.any():
            Zf.append(d["Z"][m_f])
            nombres += list(d["etiquetas"][m_f])
    if Zf:
        Zf = np.vstack(Zf)
        prob = tflite_probabilities(blob, std(Zf))
        conf = prob.max(axis=1)
        bajo = float((conf < C.CONFIDENCE_FLOOR).mean())
        print(f"\n  {len(Zf)} ventanas de reposo/apnea")
        print(f"  confianza mediana {np.median(conf):.3f}, "
              f"por debajo del piso {C.CONFIDENCE_FLOOR}: {bajo:.0%}")
        if bajo < 0.5:
            print("\n  AVISO: el dispositivo emitiria un veredicto en la mayoria de")
            print("  las ventanas de reposo. No hay clase para 'respiracion normal',")
            print("  asi que el piso de confianza es lo unico que lo frena, y aqui")
            print("  se mide que no basta.")
    print("\n" + "=" * 74)
    print(f"  artefactos en {args.out.relative_to(REPO)}")
    print("  siguiente: python ml/scripts/emit_c_artifacts.py")
    print("=" * 74)

    (args.out / "report.json").write_text(json.dumps({
        "fuente": "grabaciones reales, 4 sesiones de s01",
        "binaria_cruzando_protocolos": v["acc"],
        "ic95": list(v["ic"]),
        "p": v["p"],
        "n_efectivo": v["n_efectivo"],
        "por_grupo": v["por_grupo"],
        "n_ventanas_entrenamiento": int(len(Z)),
        "bytes_modelo": len(blob),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
