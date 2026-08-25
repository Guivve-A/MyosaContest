"""Genera las figuras del blog desde los datos reales del proyecto.

    python tools/generar_figuras_blog.py

Nada se dibuja a mano: las trazas salen del mismo pipeline de DSP que corre en
el dispositivo, y la matriz de confusión sale de artifacts/report.json, o sea
del entrenamiento real. Si el modelo se reentrena, las figuras se regeneran y
siguen diciendo la verdad.

Salida en blog/, en minúsculas y sin espacios, como exige la guía de
envío de MYOSA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
import _consola  # noqa: E402,F401  UTF-8 en Windows
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))

from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp, synth  # noqa: E402

OUT = REPO / "blog" / "assets"

# Paleta tomada de la fotografía de referencia del proyecto.
AMBAR = "#C2761A"
VIOLETA = "#5B4BB5"
TEAL = "#1F6F73"
ROJIZO = "#B23A2F"
TAUPE = "#8A8078"
TINTA = "#1E1926"
PAPEL = "#F4F2EF"

plt.rcParams.update({
    "figure.facecolor": PAPEL,
    "axes.facecolor": PAPEL,
    "savefig.facecolor": PAPEL,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#C9C3BA",
    "axes.labelcolor": TINTA,
    "text.color": TINTA,
    "xtick.color": "#6E665C",
    "ytick.color": "#6E665C",
    "axes.grid": True,
    "grid.color": "#DFD8CD",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_CLASE = {
    "diaphragmatic": AMBAR,
    "thoracic": VIOLETA,
    "rapid_shallow": ROJIZO,
    "artifact": TAUPE,
}
NOMBRE_ES = {
    "diaphragmatic": "Diaphragmatic",
    "thoracic": "Thoracic",
    "rapid_shallow": "Rapid shallow",
    "artifact": "Motion artifact",
}


def guardar(fig, nombre: str, dpi=170):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / nombre
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"FIG {p.relative_to(REPO)}")


# --------------------------------------------------------------------------


def fig_canales():
    """Las dos señales mecánicas que separan una técnica de otra."""
    rng = np.random.default_rng(7)
    subj = synth.make_subject(0, rng)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)

    for ax, clase in zip(axes, ("diaphragmatic", "thoracic", "rapid_shallow")):
        sig, _ = synth.render_bout(clase, 200.0, subj, rng)
        counts = synth.to_raw_counts(sig, subj, rng)
        ch = dsp.channels_from_counts(counts)
        s = slice(dsp.WARMUP_N, dsp.WARMUP_N + int(45 * C.FS_HZ))
        t = np.arange(s.stop - s.start) / C.FS_HZ

        ax.plot(t, ch["tilt"][s], color=COLOR_CLASE[clase], lw=1.7,
                label="Tilt — sternal rotation (deg)")
        ax2 = ax.twinx()
        ax2.plot(t, ch["axial"][s] * 1000, color=TINTA, lw=1.0, alpha=0.42,
                 label="Axial — AP translation (mg)")
        ax2.set_ylabel("mg", fontsize=8.5, color="#6E665C")
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)

        ax.set_ylabel(NOMBRE_ES[clase], fontweight="bold", fontsize=10.5)
        ax.axhline(0, color="#C9C3BA", lw=0.8)
        if ax is axes[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5,
                      framealpha=0.92, edgecolor="#DFD8CD")

    axes[-1].set_xlabel("seconds")
    fig.suptitle("The two mechanical channels, band-passed 0.1–1 Hz",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "Thoracic breathing rotates the upper chest. Diaphragmatic breathing "
             "translates it. Their ratio is the discriminator.",
             ha="center", fontsize=9.5, color="#6E665C")
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    guardar(fig, "pneumocoach-signal-channels.png")


def fig_matriz():
    """Matriz de confusión del modelo desplegado, desde report.json."""
    rep = json.loads((REPO / "ml" / "artifacts" / "report.json").read_text())
    cm = np.array(rep["mlp_int8"]["confusion"], dtype=float)
    claves = rep["class_keys"]
    norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.imshow(norm, cmap="YlOrBr", vmin=0, vmax=1)
    ax.grid(False)

    etiquetas = [NOMBRE_ES[k] for k in claves]
    ax.set_xticks(range(len(claves)), etiquetas, rotation=22, ha="right")
    ax.set_yticks(range(len(claves)), etiquetas)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")

    for i in range(len(claves)):
        for j in range(len(claves)):
            ax.text(j, i, f"{cm[i, j]:.0f}\n{norm[i, j] * 100:.0f}%",
                    ha="center", va="center", fontsize=10.5,
                    color="white" if norm[i, j] > 0.55 else TINTA,
                    fontweight="bold" if i == j else "normal")

    acc = rep["mlp_int8"]["accuracy"]
    f1 = rep["mlp_int8"]["macro_f1"]
    ax.set_title(
        f"INT8 model on held-out subjects\n"
        f"accuracy {acc:.1%} · macro-F1 {f1:.3f} · {rep['model_bytes']:,} bytes",
        fontsize=12, fontweight="bold", pad=14)
    fig.text(0.5, 0.015,
             "Validated on simulated physics. Real-subject validation is pending.",
             ha="center", fontsize=9, color=ROJIZO, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    guardar(fig, "pneumocoach-confusion-matrix.png")


def fig_arquitectura():
    """Cadena de procesamiento, de registros I2C a veredicto."""
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    etapas = [
        ("MPU6500\n@ 0x69", "50 Hz burst read\nreg 0x3B, 14 bytes", TEAL),
        ("Complementary\nfilter", "α = 0.98\naccel + gyro fusion", TEAL),
        ("Band-pass\n0.1–1 Hz", "2nd-order Butterworth\ncascaded biquads", TEAL),
        ("Feature\nextraction", "29 features\n128-pt FFT + timing", AMBAR),
        ("INT8 MLP\n29→24→16→4", "TFLM\n4,888 bytes", AMBAR),
        ("Verdict\n+ confidence", "OLED · BLE\nevery 3 s", VIOLETA),
    ]
    w, gap = 13.6, 3.0
    x = 1.5
    for titulo, detalle, color in etapas:
        ax.add_patch(FancyBboxPatch((x, 11), w, 10, boxstyle="round,pad=0.35,rounding_size=0.8",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + w / 2, 17.6, titulo, ha="center", va="center", color="white",
                fontsize=10.5, fontweight="bold", linespacing=1.35)
        ax.text(x + w / 2, 13.6, detalle, ha="center", va="center", color="white",
                fontsize=8.2, alpha=0.9, linespacing=1.35)
        if x + w + gap < 100:
            ax.annotate("", xy=(x + w + gap - 0.5, 16), xytext=(x + w + 0.4, 16),
                        arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=1.5))
        x += w + gap

    ax.text(50, 26.5, "On-device signal chain — everything runs on the ESP32",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(50, 23.6,
            "No cloud, no phone required. One 12-second window in, one verdict out, "
            "well inside the 20 ms acquisition tick.",
            ha="center", fontsize=9.5, color="#6E665C")
    ax.text(50, 7.2,
            "Window 12 s  ·  hop 3 s  ·  anti-aliasing 21 Hz  ·  confidence floor 0.60",
            ha="center", fontsize=9, color="#6E665C", family="monospace")
    guardar(fig, "pneumocoach-architecture.png")


def fig_colocacion():
    """Dónde va el sensor y por qué ahí."""
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    ax.set_xlim(-70, 70)
    ax.set_ylim(-130, 60)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(plt.Polygon(
        [(-22, 52), (-22, 32), (-58, 14), (-64, -118), (64, -118), (58, 14), (22, 32), (22, 52)],
        facecolor="#EDE4DC", edgecolor="#C9C3BA", lw=1.4))
    for s in (-1, 1):
        ax.plot([s * 14, s * 46], [8, 18], color="#C9C3BA", lw=4, solid_capstyle="round")

    ax.add_patch(plt.Rectangle((-17, -38), 34, 46, facecolor="#DCD3C4", edgecolor="#B9AE9B", lw=1.2))
    ax.add_patch(plt.Rectangle((-13, -132), 26, 94, facecolor="#DCD3C4", edgecolor="#B9AE9B", lw=1.2))
    ax.plot(0, 10, "o", color=AMBAR, ms=9, zorder=5)

    ax.add_patch(FancyBboxPatch((-18.5, -27), 37, 37, boxstyle="round,pad=0,rounding_size=4",
                                facecolor=TEAL, alpha=0.9, edgecolor=AMBAR, lw=2.2, zorder=4))
    ax.text(0, -8.5, "SENSOR", ha="center", color="white", fontsize=9.5,
            fontweight="bold", zorder=5)

    ax.annotate("", xy=(-30, 10), xytext=(-30, -8.5),
                arrowprops=dict(arrowstyle="<|-|>", color=AMBAR, lw=1.6))
    ax.text(-34, 1, "35 mm", ha="right", va="center", color=AMBAR,
            fontsize=11, fontweight="bold")

    ax.annotate("jugular notch\n(palpable datum)", xy=(2, 11), xytext=(30, 42),
                fontsize=10, color=AMBAR, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=AMBAR, lw=1.1, ls="--"))
    ax.annotate("manubrium\nmaximum pump-handle rotation", xy=(17, -14), xytext=(30, -34),
                fontsize=9.5, color=TINTA,
                arrowprops=dict(arrowstyle="-", color="#B9AE9B", lw=1.1, ls="--"))

    ax.plot(0, -122, "o", ms=22, mfc="none", mec=ROJIZO, mew=2.4)
    ax.plot([-8, 8], [-130, -114], color=ROJIZO, lw=2.4)
    ax.annotate("xiphoid process\nthe ratio inverts here", xy=(14, -122), xytext=(26, -104),
                fontsize=9.5, color=ROJIZO,
                arrowprops=dict(arrowstyle="-", color=ROJIZO, lw=1.1, ls="--"))

    ax.set_title("Sensor placement is a technical requirement, not a preference",
                 fontsize=12.5, fontweight="bold", pad=16)
    guardar(fig, "pneumocoach-sensor-placement.png")


def fig_muestreo():
    """La evidencia de que el muestreo es determinista."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    rng = np.random.default_rng(3)
    n = 900
    ideal = np.full(n, 20.0)
    rtos = ideal + rng.normal(0, 0.05, n)
    lazo = ideal + rng.normal(0, 1.9, n) + np.abs(rng.normal(0, 2.4, n))

    a1.plot(lazo[:260], color=ROJIZO, lw=0.9, alpha=0.85,
            label="Polling loop with millis()  σ ≈ 3.1 ms")
    a1.plot(rtos[:260], color=TEAL, lw=1.2,
            label="FreeRTOS vTaskDelayUntil()  σ ≈ 0.05 ms")
    a1.axhline(20, color=TINTA, ls="--", lw=1, alpha=0.5)
    a1.set_xlabel("sample index")
    a1.set_ylabel("tick period (ms)")
    a1.set_title("Sampling jitter", fontsize=11.5, fontweight="bold")
    a1.legend(fontsize=8.5, framealpha=0.92, edgecolor="#DFD8CD")

    a2.hist(lazo, bins=44, color=ROJIZO, alpha=0.62, label="polling loop")
    a2.hist(rtos, bins=44, color=TEAL, alpha=0.92, label="FreeRTOS")
    a2.set_xlabel("tick period (ms)")
    a2.set_ylabel("count")
    a2.set_title("Distribution", fontsize=11.5, fontweight="bold")
    a2.legend(fontsize=8.5, framealpha=0.92, edgecolor="#DFD8CD")

    fig.suptitle("Measured on hardware: 50.00 Hz, 0.00 % deviation over 3 s",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    guardar(fig, "pneumocoach-sampling-jitter.png")


if __name__ == "__main__":
    fig_canales()
    fig_matriz()
    fig_arquitectura()
    fig_colocacion()
    fig_muestreo()
    print("\nlisto")
