"""La figura de resultados que se puede ensenar, con la cifra que se sostiene.

    python tools/figura_resultados.py

Por que existe
--------------
La figura antigua `pneumocoach-confusion-matrix.png` dice 89.7 % (cifra retirada) y
4888 bytes sobre cuatro clases, medidos contra "simulated physics": venia de un
generador sintetico que despues medimos divergiendo 15-16 sigma de los datos
reales. Esta herramienta produce su sustituta a partir del modelo que de verdad
corre en el dispositivo.

La salvedad va en la MISMA linea que la cifra y a menos de 120 caracteres de
ella. `tools/auditoria.py` la busca con un lookahead sobre el que `.` no casa
salto de linea, asi que una salvedad en la linea siguiente no cuenta y el
fichero sale marcado. Ojo tambien: la palabra que reconoce en espanol es
"retirad", no "retractad".

Que se dibuja y por que
-----------------------
Izquierda: la matriz de confusion de la pregunta BINARIA -diafragmatica contra
toracica-, agrupando los dos pliegues de dejar-un-grupo-de-protocolo-fuera.
Ningun pliegue se entrena con una sesion de su mismo protocolo, que es el error
que inflo dos veces las cifras anteriores.

Derecha: la exactitud de cada pliegue con su intervalo, y el azar. Un punto sin
su intervalo invita a comparar ruido.

El pie NO es decorativo. Dice n efectivo, un solo sujeto, y que el veredicto
solo vale durante un ejercicio guiado. Una figura que se proyecta sin su alcance
escrito encima acaba citada sin el.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "ml" / "scripts"))
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pneumocoach import config as C  # noqa: E402
from pneumocoach.train import Standardiser  # noqa: E402
import entrenar_real as ER  # noqa: E402

CREMA = "#FBF9F5"
TINTA = "#22201d"
AMBAR = "#b5651d"
GRIS = "#8c877e"
SEED = 0

# Junto al .md, no en un subdirectorio: la guia de envio de MYOSA exige que el
# markdown, las imagenes y el video esten en la MISMA carpeta.
DESTINO = REPO / "blog" / "pneumocoach-resultado-real.png"

# La figura se proyecta ante un jurado en ingles; los grupos se llaman por su
# protocolo en el repositorio y aqui se muestran traducidos. Solo cambia la
# etiqueta: el agrupamiento sigue siendo el del codigo.
EN = {"protocolo": "long protocol", "esfuerzo": "effort protocol",
      "pooled": "pooled"}


def matriz_y_pliegues():
    """Recorre los pliegues acumulando predicciones binarias reales."""
    ses = {k: ER.carga_sesion(p) for k, (p, _) in ER.SESIONES.items()}
    i_dia = C.CLASS_INDEX["diaphragmatic"]
    i_tor = C.CLASS_INDEX["thoracic"]

    cm = np.zeros((2, 2), dtype=int)
    pliegues = []
    for g in sorted({v[1] for v in ER.SESIONES.values()}):
        te = [k for k in ses if ER.SESIONES[k][1] == g]
        tr = [k for k in ses if ER.SESIONES[k][1] != g]
        Ztr, ytr = ER.a_dataset(ses, tr)
        Zte, yte = ER.a_dataset(ses, te)
        std = Standardiser.fit(Ztr)
        modelo = ER.entrena(std(Ztr), ytr, SEED)
        prob = modelo.predict(std(Zte), verbose=0)

        b = np.isin(yte, [i_dia, i_tor])
        pred = np.where(prob[b][:, i_dia] >= prob[b][:, i_tor], i_dia, i_tor)
        real = yte[b]
        for r, p in zip(real, pred):
            cm[0 if r == i_dia else 1, 0 if p == i_dia else 1] += 1

        n = int(b.sum())
        acc = float((pred == real).mean())
        n_ef = n / (C.WINDOW_N / C.HOP_N)
        lo, hi, _ = ER.ic_binomial(acc, n_ef)
        pliegues.append({"grupo": g, "acc": acc, "n": n, "n_ef": n_ef,
                         "lo": lo, "hi": hi})
    return cm, pliegues


def _p(v: float) -> str:
    """`p = 0.0000` no dice nada. Un exponente si.

    El valor medido esta en el orden de 1e-6; formatearlo con cuatro decimales
    lo convierte en un cero que un lector interpreta como redondeo o como fallo.
    """
    if v >= 1e-4:
        return f"p = {v:.4f}"
    exp = int(np.floor(np.log10(v)))
    sup = str(-exp).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"p = {v / 10 ** exp:.1f} × 10⁻{sup}"


def dibuja(cm, pliegues, glob):
    fig = plt.figure(figsize=(13.0, 7.4), dpi=170, facecolor=CREMA)
    fig.subplots_adjust(left=0.09, right=0.955, top=0.755, bottom=0.275,
                        wspace=0.42)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])

    # ---- matriz -----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0], facecolor=CREMA)
    prop = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(prop, cmap="copper_r", vmin=0.0, vmax=1.0)
    etiquetas = ["Diaphragmatic", "Thoracic"]
    ax.set_xticks([0, 1], etiquetas, fontsize=11)
    ax.set_yticks([0, 1], etiquetas, fontsize=11, rotation=90, va="center")
    ax.set_xlabel("Predicted", fontsize=11, color=TINTA, labelpad=8)
    ax.set_ylabel("True", fontsize=11, color=TINTA, labelpad=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{prop[i, j]:.0%}", ha="center",
                    va="center", fontsize=15, weight="bold",
                    color="white" if prop[i, j] > 0.55 else TINTA)
    ax.set_title("Leave-one-protocol-group-out", fontsize=12, color=TINTA,
                 pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=GRIS)

    # ---- pliegues ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor=CREMA)
    y = np.arange(len(pliegues) + 1)
    filas = pliegues + [{"grupo": "pooled", "acc": glob["acc"],
                         "lo": glob["ic"][0], "hi": glob["ic"][1],
                         "n_ef": glob["n_efectivo"]}]
    for k, f in enumerate(filas):
        destacado = f["grupo"] == "pooled"
        col = AMBAR if destacado else GRIS
        ax2.plot([f["lo"], f["hi"]], [y[k], y[k]], color=col,
                 lw=5 if destacado else 3, solid_capstyle="round",
                 alpha=0.9 if destacado else 0.55)
        ax2.plot([f["acc"]], [y[k]], "o", color=col,
                 ms=11 if destacado else 8, zorder=3)
        ax2.text(f["hi"] + 0.015, y[k], f"{f['acc']:.3f}", va="center",
                 fontsize=12 if destacado else 11, color=TINTA,
                 weight="bold" if destacado else "normal")
    ax2.axvline(0.5, color=TINTA, ls=":", lw=1.4, alpha=0.6)
    ax2.text(0.5, -0.62, " chance", fontsize=10, color=GRIS, va="center")
    ax2.set_yticks(
        y, [f"tested on\n{EN.get(f['grupo'], f['grupo'])}  ·  "
            f"n_eff {f['n_ef']:.0f}" for f in filas],
        fontsize=10, color=TINTA)
    ax2.set_ylim(-0.85, len(filas) - 0.4)
    ax2.set_xlim(0.30, 1.05)
    ax2.set_xlabel("Binary accuracy, 95 % Jeffreys interval", fontsize=11,
                   color=TINTA, labelpad=8)
    ax2.grid(axis="x", color=GRIS, alpha=0.18)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0, colors=GRIS)

    # ---- titulo y pie -----------------------------------------------------
    fig.text(0.5, 0.955, "Diaphragmatic vs thoracic, measured on a real chest",
             ha="center", fontsize=17, weight="bold", color=TINTA)
    fig.text(0.5, 0.895,
             f"{glob['acc']:.3f}   ·   95 % CI [{glob['ic'][0]:.3f}, "
             f"{glob['ic'][1]:.3f}]   ·   {_p(glob['p'])}   ·   "
             f"n_eff {glob['n_efectivo']:.0f}   ·   "
             f"INT8 model, 5,448 bytes on the ESP32",
             ha="center", fontsize=11.5, color=AMBAR)
    fig.text(0.5, 0.055,
             "Four sessions of ONE subject. No fold is trained on a session "
             "sharing its protocol.\n"
             "This measures stability across sessions and protocols — NOT "
             "generalisation across people, which remains open.\n"
             "The verdict is only valid during a guided exercise: the device "
             "cannot tell whether the patient is exercising at all.",
             ha="center", fontsize=10, color=GRIS, linespacing=1.7)
    return fig


def main() -> int:
    print("Recorriendo los pliegues (esto entrena, tarda)...")
    cm, pliegues = matriz_y_pliegues()
    ses = {k: ER.carga_sesion(p) for k, (p, _) in ER.SESIONES.items()}
    glob = ER.valida_por_grupo(ses, SEED)

    print()
    print(f"  binaria agrupada  {glob['acc']:.3f}")
    print(f"  IC 95 %           [{glob['ic'][0]:.3f}, {glob['ic'][1]:.3f}]")
    print(f"  p                 {glob['p']:.5f}")
    print(f"  n efectivo        {glob['n_efectivo']:.0f}")
    for f in pliegues:
        print(f"  pliegue {f['grupo']:<10} {f['acc']:.3f}  "
              f"[{f['lo']:.3f}, {f['hi']:.3f}]  n={f['n']}")
    print()

    # La cifra que se dibuja tiene que ser la que dice el repositorio. Si el
    # modelo cambia y nadie actualiza docs/TECHNICAL_STATUS.md, esto lo detiene
    # aqui en vez de dejar que salga proyectado en una pantalla.
    assert abs(glob["acc"] - 0.750) < 0.06, (
        f"La binaria salio {glob['acc']:.3f}; el repositorio afirma 0.750. "
        "Reconciliar antes de generar la figura.")

    fig = dibuja(cm, pliegues, glob)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINO, facecolor=CREMA)
    print(f"Escrita {DESTINO.relative_to(REPO)} "
          f"({DESTINO.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
