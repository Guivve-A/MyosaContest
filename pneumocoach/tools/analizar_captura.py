"""Pasa una captura real por el mismo DSP con el que se entreno el modelo.

    python tools/analizar_captura.py data/raw/s01_protocolo_20260803_161500.csv

Hace tres cosas, en orden de importancia:

1. **Prueba visual.** Grafica la senal filtrada por bloque etiquetado. Si el ojo
   humano no distingue una respiracion diafragmatica de una toracica en esa
   grafica, el modelo tampoco va a poder. Es el filtro mas barato que existe y
   se aplica antes de entrenar nada.

2. **Veredicto sobre la premisa.** Calcula `log_tilt_axial_ratio` por bloque y
   mide si separa diafragmatica de toracica en datos REALES. Toda la
   discriminacion del proyecto descansa en eso y hasta ahora solo se ha
   verificado contra fisica que simulamos nosotros mismos (ver ADR-0003).

3. **Real contra sintetico.** Compara la distribucion de cada caracteristica
   entre la captura y el generador. Una divergencia grande significa que el
   modelo entrenado en simulacion no va a transferir, y hay que saberlo ahora.

Salida: un PNG por seccion en data/analisis/ y un resumen por consola.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
import _consola  # noqa: E402,F401  UTF-8 en Windows
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))

from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402

SALIDA = REPO / "data" / "analisis"

# Mapea las etiquetas del protocolo de captura a las clases del modelo.
# `reposo` y `apnea` no son clases del modelo actual: se analizan igual pero
# no se comparan contra el sintetico.
A_CLASE = {
    "diaphragmatic": "diaphragmatic",
    "thoracic": "thoracic",
    "rapid_shallow": "rapid_shallow",
    "artifact": "artifact",
}


def cargar(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (cuentas int16 (n,6), etiquetas (n,) str)."""
    filas, etiquetas = [], []
    with path.open(encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#") or linea.startswith("seq,"):
                continue
            p = linea.split(",")
            if len(p) < 10:
                continue
            try:
                filas.append([int(p[2]), int(p[3]), int(p[4]),
                              int(p[5]), int(p[6]), int(p[7])])
            except ValueError:
                continue
            etiquetas.append(p[9])
    if not filas:
        sys.exit(f"No se leyo ninguna muestra de {path}")
    return np.asarray(filas, dtype=np.int16), np.asarray(etiquetas)


def integridad(counts: np.ndarray, path: Path) -> None:
    print("\n--- integridad de la senal ---")
    n = len(counts)
    print(f"  muestras: {n}  ({n / C.FS_HZ:.1f} s a {C.FS_HZ:.0f} Hz)")

    sat = int(np.count_nonzero(np.abs(counts[:, :3]) >= 32700))
    if sat:
        print(f"  AVISO: {sat} muestras de acelerometro en saturacion")
    ceros = int(np.count_nonzero(np.all(counts == 0, axis=1)))
    if ceros:
        print(f"  AVISO: {ceros} muestras todas en cero (bus caido?)")

    g = np.linalg.norm(counts[:, :3].astype(float) / C.ACCEL_LSB_PER_G, axis=1)
    print(f"  |accel| medio: {g.mean():.3f} g  (debe rondar 1.000)")
    if abs(g.mean() - 1.0) > 0.15:
        print("  AVISO: la magnitud no ronda 1 g. Revisar escala o montaje.")

    ruido = float(np.std(np.diff(counts[:, 2].astype(float) / C.ACCEL_LSB_PER_G)))
    print(f"  ruido muestra a muestra en az: {ruido * 1000:.2f} mg")
    if ruido * 1000 > 20:
        print("  AVISO: ruido alto. Cable suelto, o el sujeto no estaba quieto.")


def bloques(etiquetas: np.ndarray) -> list[tuple[str, int, int]]:
    """Agrupa muestras consecutivas con la misma etiqueta."""
    out, ini = [], 0
    for i in range(1, len(etiquetas) + 1):
        if i == len(etiquetas) or etiquetas[i] != etiquetas[ini]:
            out.append((str(etiquetas[ini]), ini, i))
            ini = i
    return out


def prueba_visual(ch: dict, etiquetas: np.ndarray, path: Path, dpi: int = 125) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segs = [b for b in bloques(etiquetas) if b[2] - b[1] > int(20 * C.FS_HZ)]
    if not segs:
        print("\n  Ningun bloque dura lo suficiente para graficar.")
        return

    # A 300 dpi la figura va a calidad de publicacion; se agranda tambien el
    # tamano fisico para que la tipografia no quede microscopica al imprimir.
    alto = 2.45 if dpi >= 250 else 2.1
    fig, axes = plt.subplots(len(segs), 1, figsize=(13, alto * len(segs)),
                             sharex=False, constrained_layout=True)
    if len(segs) == 1:
        axes = [axes]

    for ax, (tec, a, b) in zip(axes, segs):
        # Ventana de 40 s del centro del bloque, ya pasado el transitorio.
        largo = min(b - a, int(40 * C.FS_HZ))
        c = (a + b) // 2
        s = slice(max(a, c - largo // 2), min(b, c + largo // 2))
        t = np.arange(s.stop - s.start) / C.FS_HZ

        ax.plot(t, ch["tilt"][s], lw=1.3, color="#00767B", label="inclinacion (grados)")
        ax2 = ax.twinx()
        ax2.plot(t, ch["axial"][s] * 1000, lw=1.0, color="#B23A2F",
                 alpha=0.75, label="axial (mg)")
        ax.set_ylabel(tec, fontsize=9, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        if ax is axes[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("segundos")
    fig.suptitle(f"Prueba visual - {path.stem}\n"
                 "Si no distingues las tecnicas a ojo, el modelo tampoco",
                 fontsize=11)
    SALIDA.mkdir(parents=True, exist_ok=True)
    out = SALIDA / f"{path.stem}_visual.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"\n  grafica -> {out.relative_to(REPO)}")


def veredicto_premisa(ch: dict, etiquetas: np.ndarray) -> None:
    """La pregunta que decide si el proyecto tiene fundamento."""
    print("\n" + "=" * 68)
    print("  VEREDICTO SOBRE LA PREMISA (ADR-0003)")
    print("=" * 68)

    j = C.FEATURE_NAMES.index("log_tilt_axial_ratio")
    por_clase: dict[str, list[float]] = {}

    for tec, a, b in bloques(etiquetas):
        if b - a < C.WINDOW_N:
            continue
        for ini in range(a, b - C.WINDOW_N + 1, C.HOP_N):
            w = {k: v[ini:ini + C.WINDOW_N] for k, v in ch.items()}
            f = dsp.extract_features(w)
            por_clase.setdefault(tec, []).append(float(f[j]))

    if not por_clase:
        print("  Sin ventanas suficientes. Grabar bloques mas largos.")
        return

    print(f"\n  {'clase':<18}{'n':>5}{'log(tilt/axial)':>20}{'desv':>10}")
    for tec, vals in sorted(por_clase.items()):
        v = np.asarray(vals)
        print(f"  {tec:<18}{len(v):>5}{v.mean():>20.3f}{v.std():>10.3f}")

    d = por_clase.get("diaphragmatic")
    t = por_clase.get("thoracic")
    if not d or not t or len(d) < 3 or len(t) < 3:
        print("\n  Faltan bloques de diafragmatica y/o toracica para decidir.")
        return

    d, t = np.asarray(d), np.asarray(t)
    pooled = np.sqrt(0.5 * (d.var() + t.var())) + 1e-9
    cohen = (t.mean() - d.mean()) / pooled

    print(f"\n  Cohen's d (toracica - diafragmatica) = {cohen:+.2f}")
    print("  La premisa predice un valor POSITIVO y grande: la respiracion")
    print("  toracica deberia ser mas dominada por rotacion.\n")

    if cohen > 1.5:
        print("  >> GO. La premisa se sostiene en datos reales.")
    elif cohen > 0.8:
        print("  >> GO CON RESERVAS. Separa, pero mas debil que en simulacion.")
        print("     Hacen falta mas sujetos antes de confiar en el numero.")
    elif cohen > 0.2:
        print("  >> ALERTA. Separacion marginal. Revisar colocacion del sensor")
        print("     antes de concluir que la premisa esta mal.")
    elif cohen > -0.2:
        print("  >> NO-GO. El cociente no separa. La premisa mecanica no se")
        print("     sostiene tal como esta planteada. Hay que replantear las")
        print("     caracteristicas antes de seguir entrenando.")
    else:
        print("  >> INVERTIDO. Separa, pero al reves de lo predicho. Puede ser")
        print("     el sensor montado al reves, o la premisa mal formulada.")
        print("     Revisar orientacion antes que nada.")


def real_vs_sintetico(ch: dict, etiquetas: np.ndarray) -> None:
    from pneumocoach import synth

    print("\n" + "=" * 68)
    print("  REAL CONTRA SINTETICO")
    print("=" * 68)

    reales: dict[str, list[np.ndarray]] = {}
    for tec, a, b in bloques(etiquetas):
        if tec not in A_CLASE or b - a < C.WINDOW_N:
            continue
        for ini in range(a, b - C.WINDOW_N + 1, C.HOP_N):
            w = {k: v[ini:ini + C.WINDOW_N] for k, v in ch.items()}
            reales.setdefault(tec, []).append(dsp.extract_features(w))

    if not reales:
        print("  Sin ventanas comparables.")
        return

    rng = np.random.default_rng(0)
    sinteticos: dict[str, list[np.ndarray]] = {}
    for tec in reales:
        if tec == "artifact":
            continue
        for sid in range(6):
            subj = synth.make_subject(sid, rng)
            sig, _ = synth.render_bout(tec, 140.0, subj, rng)
            cnt = synth.to_raw_counts(sig, subj, rng)
            cs = dsp.channels_from_counts(cnt)
            s = slice(dsp.WARMUP_N, dsp.WARMUP_N + C.WINDOW_N)
            sinteticos.setdefault(tec, []).append(
                dsp.extract_features({k: v[s] for k, v in cs.items()}))

    print("\n  Divergencia por caracteristica (|media_real - media_sint| / sigma_sint)")
    print("  Valores > 2 significan que el sintetico no representa lo real.\n")

    for tec in sorted(set(reales) & set(sinteticos)):
        R = np.stack(reales[tec])
        S = np.stack(sinteticos[tec])
        div = np.abs(R.mean(0) - S.mean(0)) / (S.std(0) + 1e-9)
        peor = np.argsort(div)[::-1][:5]
        print(f"  {tec}  (n_real={len(R)})   divergencia mediana {np.median(div):.2f}")
        for k in peor:
            print(f"      {C.FEATURE_NAMES[k]:<26} {div[k]:>6.2f}")
        print()

    print("  Si la divergencia mediana supera 2, reentrenar sobre datos reales")
    print("  y usar el sintetico solo como pre-entrenamiento.")


def cargar_calibracion(
    sujeto: str, fecha_sesion: str | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Matriz de montaje que corresponde a ESTA sesion, no la ultima medida.

    La rotacion del sensor sobre el torso solo vale mientras el sensor no se
    despegue. Cada vez que se recoloca hay que volver a medirla, asi que hay
    varias matrices por sujeto y elegir la equivocada corrompe el analisis en
    silencio: los canales salen plausibles y la fisica es otra.

    Convencion de archivo:
        {sujeto}.json                      la vigente
        {sujeto}_montaje_YYYYMMDD.json     las anteriores, archivadas

    Se elige la mas reciente cuya fecha de medicion sea ANTERIOR o igual al
    inicio de la grabacion. Sin `fecha_sesion` se devuelve la vigente, que es
    el comportamiento util para medir en vivo.
    """
    import json
    import datetime as _dt

    carpeta = REPO / "data" / "calibracion"
    candidatos = []
    for q in sorted(carpeta.glob(f"{sujeto}*.json")):
        try:
            d = json.loads(q.read_text(encoding="utf-8"))
            candidatos.append((d.get("fecha_utc", ""), q, d))
        except Exception:
            continue
    if not candidatos:
        return None, None

    if fecha_sesion:
        previas = [c for c in candidatos if c[0] and c[0] <= fecha_sesion]
        if previas:
            _, p, d = max(previas, key=lambda c: c[0])
        else:
            # Ninguna medicion previa a la grabacion. Se usa la mas antigua,
            # pero hay que decirlo: puede no corresponder.
            _, p, d = min(candidatos, key=lambda c: c[0])
            print(f"\nAVISO: no hay calibracion de montaje anterior a "
                  f"{fecha_sesion[:10]}.")
            print(f"         Se usa {p.name}, que es POSTERIOR a la grabacion.")
    else:
        p = carpeta / f"{sujeto}.json"
        if not p.exists():
            _, p, d = max(candidatos, key=lambda c: c[0])
        else:
            d = json.loads(p.read_text(encoding="utf-8"))

    R = np.asarray(d["R_sensor_a_cuerpo"], dtype=np.float64)
    sesgo = np.asarray(d.get("sesgo_giro_dps", [0, 0, 0]), dtype=np.float64)
    print(f"\ncalibracion de montaje: data/calibracion/{p.name}"
          f"  ({d.get('fecha_utc', '?')[:16]})")
    print(f"    giro respecto al marco ideal: {d.get('giro_vs_marco_ideal_deg', '?')} grados")
    err = d.get("error_perpendicular_deg")
    if err is not None and err > 12:
        print(f"    AVISO: error de perpendicular {err} grados. La calibracion")
        print("           es imperfecta y `axial` lleva algo de `tilt` mezclado.")
    return R, sesgo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--sujeto", help="para cargar data/calibracion/<sujeto>.json; "
                                     "por defecto se deduce del nombre del archivo")
    ap.add_argument("--sin-calibracion", action="store_true",
                    help="ignorar la matriz de montaje (para comparar)")
    ap.add_argument("--sin-graficas", action="store_true")
    ap.add_argument("--dpi", type=int, default=125,
                    help="300 para calidad de publicacion")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"No existe {args.csv}")

    counts, etiquetas = cargar(args.csv)
    print("=" * 68)
    print(f"  {args.csv.name}")
    print("=" * 68)
    integridad(counts, args.csv)

    R = sesgo = None
    if not args.sin_calibracion:
        sujeto = args.sujeto or args.csv.stem.split("_")[0]
        R, sesgo = cargar_calibracion(sujeto)
        if R is None:
            print(f"\n  SIN CALIBRACION DE MONTAJE para '{sujeto}'.")
            print("  Se asume que el sensor quedo alineado con el marco anatomico,")
            print("  cosa que casi nunca pasa sobre un torso real. Correr primero:")
            print(f"    python tools/orientacion.py --puerto COM9 --sujeto {sujeto}")

    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)

    print("\n--- bloques ---")
    for tec, a, b in bloques(etiquetas):
        print(f"  {tec:<18}{(b - a) / C.FS_HZ:>7.1f} s   ({b - a} muestras)")

    if not args.sin_graficas:
        prueba_visual(ch, etiquetas, args.csv, dpi=args.dpi)

    veredicto_premisa(ch, etiquetas)
    real_vs_sintetico(ch, etiquetas)


if __name__ == "__main__":
    main()
