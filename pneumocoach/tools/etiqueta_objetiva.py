"""Etiqueta medida, no dicha: indices toraco-abdominales por respiracion.

    python tools/etiqueta_objetiva.py --autoprueba
    python tools/etiqueta_objetiva.py data/raw/<sesion>.csv --abdomen <alineado.npy>

El problema que resuelve
------------------------
ADR-0007: las etiquetas de este proyecto registran lo que el sujeto CREIA estar
haciendo. Dos sesiones separadas 30 minutos invirtieron la relacion mecanica
entre clases porque cambio la consigna, no la fisiologia. Mientras la etiqueta
salga de una instruccion, ningun volumen de datos ayuda.

Un segundo acelerometro sobre el abdomen -el de un telefono sirve- convierte la
etiqueta en una medicion. El dispositivo desplegado sigue usando UN sensor: el
segundo existe solo para construir el dataset, y por eso no viola la
restriccion del kit MYOSA.

La matematica
-------------
Konno-Mead enfrenta el movimiento de la caja toracica (RC) contra el del
abdomen (AB). De cada respiracion salen dos numeros que no dependen de la
intencion del sujeto:

    phi   = fase(X_rc * conj(X_ab))     en la fundamental de ESA respiracion
    rc_ix = log10(D_rc / D_ab)          cociente de amplitudes

`phi` cerca de 0 es sincronia; hacia +/-180 grados es movimiento paradojico. Es
el indice estandar de asincronia toraco-abdominal, es adimensional y no hace
falta calibrar nada para compararlo entre sujetos.

Como se estima la fundamental
-----------------------------
La ventana de analisis es EXACTAMENTE una respiracion, remuestreada a N puntos
entre sus dos fronteras. Con eso la fundamental cae justo en el bin 1 de la DFT
de N puntos, no hay fuga espectral y basta un unico coeficiente complejo por
canal. Es mas limpio que buscar un pico en un espectro de una sola respiracion,
donde la resolucion no da.

Las fronteras salen SIEMPRE del canal esternal, que es el fuerte (SNR ~100
frente a ~1 del axial), y se aplican tal cual al abdominal. Segmentar cada
canal por su cuenta emparejaria la respiracion k de uno con la k del otro y se
desalinearia en cuanto cualquiera de los dos detectores se saltara un ciclo.

Sobre las unidades, que es donde esto se podria falsear
-------------------------------------------------------
`rc_ix` NO es el RC% clinico y no se presenta como tal. El RC% de la literatura
es una fraccion del volumen corriente, y para calcularlo harian falta dos
sensores con la misma funcion de transferencia de volumen a senal. No la
tienen: distinto sitio anatomico, distinta masa acoplada, y en el esternon
medimos rotacion mientras en el abdomen medimos traslacion.

Lo que si es legitimo:

  - Dentro de un sujeto y una colocacion, `rc_ix` es un eje MONOTONO. Su cero
    es arbitrario; su orden no. Y el orden es lo unico que las pruebas de este
    plan necesitan.
  - Se convierte aceleracion a desplazamiento dividiendo por omega^2 antes de
    hacer el cociente. Con senal de banda estrecha eso es exacto, y sin ese
    paso el cociente dependeria de la frecuencia respiratoria — es decir,
    `rapid_shallow` saldria separada por un artefacto de unidades y no por su
    mecanica.
  - `phi` no necesita ninguna de estas salvedades. Por eso es el indice
    primario y `rc_ix` el secundario.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401
from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402

#: Amplitud minima, en unidades del canal ya filtrado, por debajo de la cual la
#: fase carece de sentido. No es un umbral inventado: se calibra contra los
#: bloques de apnea, donde por construccion no hay respiracion. Ver --autoprueba.
AMPLITUD_NULA = 1e-9


def coef_fundamental(x: np.ndarray, s0: float, s2: float) -> complex:
    """Coeficiente complejo de la fundamental sobre UNA respiracion.

    `s0` y `s2` vienen en muestras fraccionarias. Se remuestrea el tramo a N
    puntos que cubren exactamente un periodo, asi que el bin 1 de la DFT de N
    puntos ES la fundamental: sin ventana, sin fuga y sin buscar picos.

    Devuelve la amplitud como modulo (la escala 2/N deja |X| en amplitud de
    pico, no en energia) y la fase en su argumento.
    """
    n = int(round(s2 - s0))
    if n < 8:
        return 0j
    # endpoint=False: N puntos que cubren un periodo sin repetir el extremo.
    t = np.linspace(s0, s2, n, endpoint=False)
    seg = np.interp(t, np.arange(len(x), dtype=float), x)
    seg = seg - seg.mean()
    k = np.arange(n)
    return complex((2.0 / n) * np.sum(seg * np.exp(-2j * np.pi * k / n)))


def indices_respiracion(
    rc: np.ndarray, ab: np.ndarray, fs: float = C.FS_HZ
) -> list[dict]:
    """Un registro por respiracion completa detectada en el canal esternal."""
    fuera: list[dict] = []
    for s0, s2, periodo in dsp.breath_bounds(rc, fs):
        if periodo <= 0:
            continue
        f0 = 1.0 / periodo
        w = 2.0 * np.pi * f0

        X_rc = coef_fundamental(rc, s0, s2)
        X_ab = coef_fundamental(ab, s0, s2)
        a_rc, a_ab = abs(X_rc), abs(X_ab)

        # Aceleracion -> desplazamiento. Exacto en banda estrecha, y sin esto
        # el cociente dependeria de la frecuencia y `rapid_shallow` se separaria
        # por unidades en vez de por mecanica.
        d_rc, d_ab = a_rc / (w * w), a_ab / (w * w)

        valido = a_rc > AMPLITUD_NULA and a_ab > AMPLITUD_NULA
        fuera.append({
            "inicio_s": s0 / fs,
            "periodo_s": periodo,
            "frecuencia_rpm": 60.0 * f0,
            "amp_rc": a_rc,
            "amp_ab": a_ab,
            # Amplitud mareal: el eje que separa reposo de una maniobra
            # deliberada, y el que hoy se pierde al normalizar por sesion.
            "amplitud_mareal": d_rc + d_ab,
            "rc_ix": float(np.log10(d_rc / d_ab)) if valido else float("nan"),
            "phi_grados": (float(np.degrees(np.angle(X_rc * np.conj(X_ab))))
                           if valido else float("nan")),
            "valido": bool(valido),
        })
    return fuera


# ==========================================================================
# Autoprueba: el estimador se valida contra senales de fase y cociente CONOCIDOS
# ==========================================================================

def _sintetica(f0: float, amp: float, fase_grados: float, dur_s: float,
               fs: float, ruido: float = 0.0, semilla: int = 0) -> np.ndarray:
    """Aceleracion de un movimiento sinusoidal de amplitud y fase dadas.

    Se genera la ACELERACION -de ahi el -omega^2- para que la autoprueba
    ejercite el mismo camino que los datos reales, incluida la division por
    omega^2 que hace `indices_respiracion`.
    """
    t = np.arange(int(dur_s * fs)) / fs
    w = 2.0 * np.pi * f0
    rng = np.random.default_rng(semilla)
    x = -amp * w * w * np.sin(w * t + np.radians(fase_grados))
    return x + ruido * rng.standard_normal(len(t))


def autoprueba() -> int:
    """Recupera fase y cociente conocidos. Un estimador sin esto es una opinion."""
    fs = C.FS_HZ
    print("=" * 74)
    print("  AUTOPRUEBA DEL ESTIMADOR")
    print("=" * 74)
    print()
    print("  Se sintetizan dos canales con cociente y desfase CONOCIDOS y se")
    print("  comprueba que el estimador los devuelve.")
    print()
    print(f"  {'f0 rpm':>7}{'ratio':>8}{'phi real':>10}{'phi est':>9}"
          f"{'err phi':>9}{'rc_ix real':>12}{'rc_ix est':>11}{'err':>8}")
    print("  " + "-" * 72)

    casos = [
        # f0_rpm, d_rc, d_ab, phi_grados, ruido
        (12.0, 1.0, 1.0, 0.0, 0.0),
        (12.0, 3.0, 1.0, 0.0, 0.0),
        (12.0, 1.0, 4.0, 0.0, 0.0),
        (8.0, 1.0, 1.0, 45.0, 0.0),
        (8.0, 1.0, 1.0, -30.0, 0.0),
        (30.0, 1.0, 2.0, 90.0, 0.0),
        (10.0, 2.0, 1.0, 20.0, 0.02),   # con ruido
        (10.0, 2.0, 1.0, 20.0, 0.10),   # con mas ruido
    ]
    peor_phi = peor_ix = 0.0
    for rpm, d_rc, d_ab, phi, ruido in casos:
        f0 = rpm / 60.0
        dur = 6.0 / f0                     # seis respiraciones
        rc = _sintetica(f0, d_rc, phi, dur, fs, ruido, semilla=1)
        ab = _sintetica(f0, d_ab, 0.0, dur, fs, ruido, semilla=2)
        regs = [r for r in indices_respiracion(rc, ab, fs) if r["valido"]]
        if not regs:
            print(f"  {rpm:>7.1f}   sin respiraciones detectadas")
            peor_phi = 1e9
            continue
        phi_est = float(np.median([r["phi_grados"] for r in regs]))
        ix_est = float(np.median([r["rc_ix"] for r in regs]))
        ix_real = float(np.log10(d_rc / d_ab))
        e_phi, e_ix = abs(phi_est - phi), abs(ix_est - ix_real)
        peor_phi, peor_ix = max(peor_phi, e_phi), max(peor_ix, e_ix)
        print(f"  {rpm:>7.1f}{d_rc / d_ab:>8.2f}{phi:>10.1f}{phi_est:>9.1f}"
              f"{e_phi:>9.2f}{ix_real:>12.3f}{ix_est:>11.3f}{e_ix:>8.3f}")

    print("  " + "-" * 72)
    print(f"  peor error de fase    : {peor_phi:.2f} grados")
    print(f"  peor error de cociente: {peor_ix:.3f} decadas")
    print()

    # Umbrales: la fase se usa para distinguir sincronia de paradoja, donde lo
    # que importa son decenas de grados. 5 grados es holgura de sobra y a la vez
    # lo bastante estrecho para detectar un estimador roto.
    ok = peor_phi < 5.0 and peor_ix < 0.05

    # Y el caso que de verdad importa: SIN respiracion no puede salir un indice.
    # Es la prueba de cordura contra los bloques de apnea del protocolo.
    rng = np.random.default_rng(7)
    quieto_rc = 1e-12 * rng.standard_normal(int(30 * fs))
    quieto_ab = 1e-12 * rng.standard_normal(int(30 * fs))
    regs = indices_respiracion(quieto_rc, quieto_ab, fs)
    validos_en_apnea = sum(1 for r in regs if r["valido"])
    print(f"  apnea simulada: {len(regs)} respiraciones detectadas, "
          f"{validos_en_apnea} con indice valido")
    if validos_en_apnea:
        print("  FALLA: el estimador produce indices sobre una senal sin "
              "respiracion.")
        ok = False

    print()
    print("  AUTOPRUEBA " + ("PASA" if ok else "FALLA"))
    return 0 if ok else 1


# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", help="sesion MYOSA en data/raw/")
    ap.add_argument("--abdomen", help=".npy del canal abdominal ya alineado a 50 Hz")
    ap.add_argument("--autoprueba", action="store_true",
                    help="valida el estimador contra senales conocidas")
    ap.add_argument("--salida", default=None, help="CSV de salida, una fila por respiracion")
    args = ap.parse_args()

    if args.autoprueba or not args.csv:
        return autoprueba()

    if not args.abdomen:
        raise SystemExit(
            "Falta --abdomen. El canal abdominal alineado lo produce\n"
            "  python tools/capturar_dual.py --alinear <sesion>")

    from analizar_captura import bloques, cargar, cargar_calibracion

    p = REPO / args.csv
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    R, sesgo = cargar_calibracion(meta.get("sujeto", "s01"), meta.get("inicio_utc"))
    counts, et = cargar(p)
    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)
    ab = np.load(REPO / args.abdomen)
    if len(ab) != len(ch["tilt"]):
        raise SystemExit(
            f"El abdomen tiene {len(ab)} muestras y el esternon "
            f"{len(ch['tilt'])}. Alinear antes con capturar_dual.py.")

    filas = []
    for tec, a, b in bloques(et):
        regs = indices_respiracion(ch["tilt"][a:b], ab[a:b], C.FS_HZ)
        for r in regs:
            r["consigna"] = tec
            r["inicio_s"] += a / C.FS_HZ
            filas.append(r)

    destino = Path(args.salida) if args.salida else (
        REPO / "data" / "etiquetas" / (p.stem + "_respiraciones.csv"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    campos = ["consigna", "inicio_s", "periodo_s", "frecuencia_rpm",
              "amp_rc", "amp_ab", "amplitud_mareal", "rc_ix", "phi_grados",
              "valido"]
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    validas = sum(1 for f in filas if f["valido"])
    print(f"  {len(filas)} respiraciones, {validas} con indice valido")
    print(f"  escrito {destino.relative_to(REPO)}")
    print()
    print("  Siguiente: python tools/compuerta_etiquetas.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
