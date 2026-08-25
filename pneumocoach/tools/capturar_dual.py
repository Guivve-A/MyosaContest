"""Alinea el registro del telefono con el del MYOSA. Dos relojes, una senal.

    python tools/capturar_dual.py --autoprueba
    python tools/capturar_dual.py --alinear data/raw/<sesion>.csv \
                                  --telefono <phyphox.csv>

Que hace y que NO hace
----------------------
La captura del esternon no cambia: la sigue haciendo `tools/capture.py`, que ya
dicta el guion cronometrado y escribe las etiquetas. Esta herramienta solo
resuelve el problema que aparece al anadir un segundo aparato: **dos relojes
independientes**, uno en el ESP32 y otro en el telefono, que empiezan en
instantes distintos y avanzan a ritmos ligeramente distintos.

Por que tres golpes al principio Y tres al final
------------------------------------------------
Un solo evento comun da el DESFASE. No dice nada de la DERIVA: si el reloj del
telefono corre 200 ppm mas rapido, en cinco minutos se van 60 ms, que a 50 Hz
son tres muestras. Suficiente para corromper una fase.

Con seis eventos repartidos en los dos extremos se ajusta una recta

    t_telefono = alfa * t_myosa + beta

por minimos cuadrados. `beta` es el desfase, `alfa - 1` la deriva, y -esto es lo
importante- **sobran cuatro grados de libertad**, asi que el residuo del ajuste
es una medida real de si la alineacion funciono. Con dos puntos y dos incognitas
el ajuste seria exacto y el residuo siempre cero: parecerian perfectos incluso
emparejando golpes equivocados.

La guardia
----------
Si el residuo pasa de 40 ms -dos muestras a 50 Hz- la sesion se marca
INUTILIZABLE y no entra al dataset. Una etiqueta calculada sobre canales
desalineados no es una etiqueta ruidosa: es una etiqueta falsa, y ademas
sistematica, porque el desfase sesga la fase toraco-abdominal siempre en el
mismo sentido. Es peor que no tener etiqueta.
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401
from pneumocoach import config as C  # noqa: E402

#: Residuo maximo del ajuste, en segundos. Dos muestras a 50 Hz.
RESIDUO_MAX_S = 0.040

#: Golpes esperados en cada extremo.
N_GOLPES = 3


def envolvente_jerk(a: np.ndarray, fs: float) -> np.ndarray:
    """Envolvente del jerk: donde hay un golpe, hay un pico estrecho.

    Un golpe seco es un transitorio de banda ancha. La derivada lo realza
    frente a la respiracion, que vive por debajo de 1 Hz, y el valor absoluto
    lo hace independiente de como este orientado cada aparato — que es
    imprescindible porque el telefono y la placa no comparten ejes.
    """
    j = np.abs(np.diff(a, prepend=a[0]))
    # Suavizado de 60 ms: junta las oscilaciones de un mismo golpe en un pico
    # unico sin llegar a fundir dos golpes consecutivos.
    n = max(1, int(round(0.060 * fs)))
    return np.convolve(j, np.ones(n) / n, mode="same")


def detecta_golpes(env: np.ndarray, fs: float, n: int,
                   al_final: bool) -> list[float]:
    """Indices FRACCIONARIOS de los `n` golpes de un extremo del registro.

    Devuelve indices, no segundos, a proposito. El esternon muestrea a 50 Hz
    exactos y ahi indice/fs es el instante; el telefono no. Su tasa nominal es
    una media y sus marcas de tiempo reales no son uniformes, asi que convertir
    con la tasa nominal mete un error que crece a lo largo del registro — justo
    donde se mide la deriva. Cada llamante interpola el indice en SU eje de
    tiempos real.

    `fs` se sigue usando para las escalas internas (ventana de busqueda y
    separacion minima entre golpes), donde una tasa aproximada basta.

    Se busca en el 25 % inicial o final. El umbral es la mediana mas 8 MAD:
    robusto frente a la respiracion, que es la senal mayoritaria, y sin
    constantes puestas a ojo sobre la amplitud absoluta, que depende del
    aparato y de la fuerza del golpe.
    """
    ancho = max(int(0.25 * len(env)), int(20 * fs))
    tramo = env[-ancho:] if al_final else env[:ancho]
    desp = len(env) - ancho if al_final else 0

    med = float(np.median(tramo))
    mad = float(np.median(np.abs(tramo - med))) or 1e-12
    umbral = med + 8.0 * mad

    # Un golpe = un cruce por encima del umbral. El instante NO se toma con
    # argmax: un golpe es un escalon corto convolucionado con el suavizado, y
    # eso deja una MESETA PLANA de varias muestras. Sobre una meseta plana el
    # maximo lo decide el ruido, y medido daba hasta 100 ms de discrepancia
    # entre los dos registros del mismo golpe — mas que el limite entero de
    # alineacion.
    #
    # El centroide del area por encima del umbral es estable frente a la meseta
    # y frente al ruido, y ademas da resolucion SUB-MUESTRA, que a 50 Hz importa:
    # sin ella el instante ya viene cuantizado a +/-20 ms.
    sep = int(round(0.25 * fs))
    picos: list[tuple[float, float]] = []   # (peso, instante fraccionario)
    encima = tramo > umbral
    k = 0
    while k < len(tramo):
        if not encima[k]:
            k += 1
            continue
        j = k
        while j < len(tramo) and encima[j]:
            j += 1
        w = tramo[k:j] - umbral
        peso = float(w.sum())
        centro = float(k + np.sum(w * np.arange(len(w))) / peso) if peso > 0 else float(k)
        if not picos or (centro - picos[-1][1]) > sep:
            picos.append((peso, centro))
        elif peso > picos[-1][0]:
            picos[-1] = (peso, centro)
        k = j

    picos.sort(key=lambda p: -p[0])
    return [desp + c for c in sorted(c for _, c in picos[:n])]


def ajusta(t_a: list[float], t_b: list[float]) -> tuple[float, float, float]:
    """Recta t_b = alfa*t_a + beta por minimos cuadrados. Devuelve (alfa, beta, residuo)."""
    if len(t_a) != len(t_b) or len(t_a) < 3:
        return (float("nan"),) * 3
    A = np.vstack([np.asarray(t_a), np.ones(len(t_a))]).T
    (alfa, beta), *_ = np.linalg.lstsq(A, np.asarray(t_b), rcond=None)
    residuo = float(np.max(np.abs(A @ np.array([alfa, beta]) - np.asarray(t_b))))
    return float(alfa), float(beta), residuo


def lee_phyphox(ruta: Path, columnas: str | None) -> tuple[np.ndarray, np.ndarray]:
    """(t, magnitud de aceleracion) desde una exportacion de Phyphox.

    Phyphox exporta con separador y decimal segun el idioma del telefono, y con
    los nombres de columna traducidos. Se olfatea el separador y se buscan las
    columnas por posicion si el nombre no se reconoce, en vez de fallar.
    """
    texto = ruta.read_text(encoding="utf-8-sig", errors="replace")
    lineas = [l for l in texto.splitlines() if l.strip()]
    if not lineas:
        raise SystemExit(f"{ruta} esta vacio")
    sep = max([",", ";", "\t"], key=lambda s: lineas[0].count(s))

    filas = list(_csv.reader(lineas, delimiter=sep))
    cab = [c.strip().lower() for c in filas[0]]

    def num(s: str) -> float:
        s = s.strip().replace(",", ".")
        return float(s) if s else float("nan")

    if columnas:
        idx = [int(i) for i in columnas.split(",")]
    else:
        it = next((i for i, c in enumerate(cab)
                   if "time" in c or "tiempo" in c), 0)
        ejes = [i for i, c in enumerate(cab)
                if any(c.startswith(p) for p in ("acceleration", "aceleraci",
                                                 "linear", "ax", "ay", "az"))
                and i != it]
        if len(ejes) < 3:
            ejes = [i for i in range(len(cab)) if i != it][:3]
        idx = [it] + ejes[:3]
    if len(idx) < 4:
        raise SystemExit(
            f"No reconozco las columnas de {ruta.name}. Cabecera: {cab}\n"
            "Pasa --columnas t,x,y,z con los indices (base 0).")

    datos = []
    for f in filas[1:]:
        if len(f) <= max(idx):
            continue
        try:
            datos.append([num(f[i]) for i in idx])
        except ValueError:
            continue
    d = np.asarray(datos, dtype=float)
    d = d[np.isfinite(d).all(axis=1)]
    if len(d) < 100:
        raise SystemExit(f"{ruta.name}: solo {len(d)} filas utiles")
    t = d[:, 0] - d[0, 0]
    mag = np.linalg.norm(d[:, 1:4], axis=1)
    return t, mag


def alinea(csv_myosa: Path, csv_tel: Path, columnas: str | None,
           salida: Path | None) -> int:
    from analizar_captura import cargar

    counts, _ = cargar(csv_myosa)
    fs = C.FS_HZ
    a_myosa = np.linalg.norm(counts[:, 0:3].astype(float) / 16384.0, axis=1)
    env_m = envolvente_jerk(a_myosa, fs)

    t_tel, mag_tel = lee_phyphox(csv_tel, columnas)
    fs_tel = float(len(t_tel) - 1) / (t_tel[-1] - t_tel[0])
    env_t = envolvente_jerk(mag_tel, fs_tel)

    print("=" * 74)
    print("  ALINEACION DE DOS RELOJES")
    print("=" * 74)
    print()
    print(f"  esternon : {len(a_myosa):6d} muestras  {len(a_myosa)/fs:7.1f} s"
          f"  {fs:6.2f} Hz")
    print(f"  telefono : {len(t_tel):6d} muestras  {t_tel[-1]:7.1f} s"
          f"  {fs_tel:6.2f} Hz")
    print()

    # Indices -> segundos. El esternon con su reloj exacto; el telefono
    # interpolando en SUS marcas de tiempo, que no son uniformes.
    i_m = (detecta_golpes(env_m, fs, N_GOLPES, False)
           + detecta_golpes(env_m, fs, N_GOLPES, True))
    i_t = (detecta_golpes(env_t, fs_tel, N_GOLPES, False)
           + detecta_golpes(env_t, fs_tel, N_GOLPES, True))
    g_m = [i / fs for i in i_m]
    g_t = list(np.interp(i_t, np.arange(len(t_tel), dtype=float), t_tel))

    print(f"  golpes esternon: {', '.join(f'{v:.3f}' for v in g_m)}")
    print(f"  golpes telefono: {', '.join(f'{v:.3f}' for v in g_t)}")
    print()
    if len(g_m) != len(g_t):
        raise SystemExit(
            f"Detectados {len(g_m)} golpes en el esternon y {len(g_t)} en el "
            f"telefono. Deben ser {2 * N_GOLPES} en cada uno. Regrabar, o "
            "revisar que los golpes fueran secos y sobre el esternon.")

    alfa, beta, residuo = ajusta(g_m, g_t)
    deriva_ppm = (alfa - 1.0) * 1e6
    print(f"  alfa            : {alfa:.9f}   (deriva {deriva_ppm:+.0f} ppm)")
    print(f"  beta            : {beta:+.4f} s")
    print(f"  residuo maximo  : {residuo * 1000:.1f} ms"
          f"   (limite {RESIDUO_MAX_S * 1000:.0f} ms)")
    print()

    if not np.isfinite(residuo) or residuo > RESIDUO_MAX_S:
        print("  SESION INUTILIZABLE. No se escribe nada.")
        print()
        print("  Una etiqueta calculada sobre canales desalineados no es")
        print("  ruidosa: es falsa y sesgada siempre en el mismo sentido.")
        return 1

    # Remuestreo del telefono al reloj del esternon.
    t_m = np.arange(len(a_myosa)) / fs
    ab = np.interp(alfa * t_m + beta, t_tel, mag_tel)

    destino = salida or (REPO / "data" / "etiquetas"
                         / (csv_myosa.stem + "_abdomen.npy"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.save(destino, ab.astype(np.float64))

    meta = {"alfa": alfa, "beta": beta, "residuo_s": residuo,
            "deriva_ppm": deriva_ppm, "fs_telefono": fs_tel,
            "golpes_esternon": g_m, "golpes_telefono": g_t,
            "origen_telefono": str(csv_tel.name)}
    destino.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  ALINEADA. Escrito {destino.relative_to(REPO)}")
    print()
    print("  Siguiente:")
    print(f"    python tools/etiqueta_objetiva.py {csv_myosa.relative_to(REPO)} "
          f"--abdomen {destino.relative_to(REPO)}")
    return 0


def _golpe(t: np.ndarray, en: float, tau: float, amp: float) -> np.ndarray:
    """Transitorio amortiguado: un golpe seco visto por un acelerometro."""
    d = t - en
    y = np.zeros_like(t)
    m = d >= 0
    y[m] = amp * np.exp(-d[m] / tau) * np.sin(2 * np.pi * 18.0 * d[m])
    return y


def autoprueba() -> int:
    """Recupera un desfase y una deriva CONOCIDOS.

    La version facil de esta prueba -los dos canales sobre la misma rejilla y
    con la misma forma de onda- pasaba trivialmente y no probaba nada: es el
    test que pasa porque las dos partes son iguales. Aqui el telefono muestrea
    a OTRA tasa, con jitter, y el golpe le llega con otra forma y otra
    amplitud, porque el abdomen no es el esternon.
    """
    fs = 50.0
    print("=" * 74)
    print("  AUTOPRUEBA DE LA ALINEACION")
    print("=" * 74)
    print()
    print("  Telefono a otra tasa, con jitter de muestreo, y el golpe con otra")
    print("  forma y amplitud. Si pasa asi, mide algo.")
    print()
    print(f"  {'beta real':>10}{'deriva real':>13}{'beta est':>10}"
          f"{'deriva est':>12}{'residuo ms':>12}{'':>8}")
    print("  " + "-" * 68)

    ok = True
    rng = np.random.default_rng(3)
    dur = 300.0
    golpes = [2.0, 3.05, 4.1, dur - 5.0, dur - 3.95, dur - 2.9]

    for beta_real, ppm_real, fs_tel in ((0.0, 0.0, 100.0),
                                        (1.234, 0.0, 104.7),
                                        (-0.75, 200.0, 200.0),
                                        (3.10, -450.0, 96.3)):
        # --- esternon, rejilla regular a 50 Hz -------------------------------
        t_m = np.arange(int(dur * fs)) / fs
        a_m = 0.02 * np.sin(2 * np.pi * 0.2 * t_m)
        for g in golpes:
            a_m += _golpe(t_m, g, 0.035, 1.0)
        a_m += 0.002 * rng.standard_normal(len(t_m))

        # --- telefono: otra tasa, con jitter, otra forma de golpe ------------
        #
        # El telefono graba ENVOLVIENDO al esternon: arranca antes y para
        # despues. No es un detalle del test. La primera version dejaba los dos
        # registros del mismo largo y con beta = +3.1 s el ultimo golpe caia
        # FUERA del registro del telefono: se detectaban 5 golpes de 6. Es un
        # requisito del protocolo de grabacion, y esta escrito en
        # docs/DUAL_SENSOR_PROTOCOL.md.
        MARGEN = 8.0
        alfa_real = 1.0 + ppm_real * 1e-6
        n_t = int((dur + 2 * MARGEN) * fs_tel)
        t_tel = np.arange(n_t) / fs_tel - MARGEN
        t_tel += (0.25 / fs_tel) * rng.standard_normal(n_t)   # jitter
        t_tel = np.sort(t_tel)
        # Instante del mismo golpe medido por el reloj del telefono.
        t_com = (t_tel - beta_real) / alfa_real               # -> reloj esternon
        a_t = 0.03 * np.sin(2 * np.pi * 0.2 * t_com + 0.4)
        for g in golpes:
            a_t += _golpe(t_com, g, 0.055, 0.6)               # otra forma
        a_t += 0.004 * rng.standard_normal(n_t)

        env_m = envolvente_jerk(a_m, fs)
        env_t = envolvente_jerk(a_t, fs_tel)
        i_m = (detecta_golpes(env_m, fs, N_GOLPES, False)
               + detecta_golpes(env_m, fs, N_GOLPES, True))
        i_t = (detecta_golpes(env_t, fs_tel, N_GOLPES, False)
               + detecta_golpes(env_t, fs_tel, N_GOLPES, True))

        if len(i_m) != N_GOLPES * 2 or len(i_t) != N_GOLPES * 2:
            print(f"  deteccion fallida: {len(i_m)} / {len(i_t)} golpes")
            ok = False
            continue

        # Igual que en `alinea`: el telefono se re-referencia a su primera
        # muestra, asi que el desfase que se recupera incluye el margen con que
        # arranco antes.
        g_m = [i / fs for i in i_m]
        t_rel = t_tel - t_tel[0]
        g_t = list(np.interp(i_t, np.arange(len(t_rel), dtype=float), t_rel))
        beta_esperado = beta_real + (0.0 - t_tel[0])

        alfa, beta, res = ajusta(g_m, g_t)
        ppm = (alfa - 1.0) * 1e6
        bien = (abs(beta - beta_esperado) < 0.02 and abs(ppm - ppm_real) < 60
                and res < RESIDUO_MAX_S)
        ok &= bien
        print(f"  {beta_esperado:>10.3f}{ppm_real:>13.0f}{beta:>10.3f}"
              f"{ppm:>12.0f}{res * 1000:>12.1f}{'  ok' if bien else '  FALLA':>8}")

    print("  " + "-" * 68)
    print()

    # Y el caso que la guardia existe para atrapar: golpes mal emparejados.
    # Si esto NO se marca inutilizable, la guardia no sirve.
    g_a = [2.0, 3.0, 4.0, 295.0, 296.0, 297.0]
    g_b = [2.0, 3.0, 4.0, 295.0, 296.0, 297.5]     # un golpe descolocado 500 ms
    _, _, res_malo = ajusta(g_a, g_b)
    atrapa = res_malo > RESIDUO_MAX_S
    print(f"  golpe descolocado 500 ms -> residuo {res_malo * 1000:.0f} ms  "
          f"{'ATRAPADO' if atrapa else 'SE COLA'}")
    ok &= atrapa

    print()
    print("  AUTOPRUEBA " + ("PASA" if ok else "FALLA"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alinear", help="CSV de la sesion MYOSA")
    ap.add_argument("--telefono", help="CSV exportado por Phyphox")
    ap.add_argument("--columnas", help="indices t,x,y,z si no se reconocen")
    ap.add_argument("--salida", help="ruta del .npy de salida")
    ap.add_argument("--autoprueba", action="store_true")
    args = ap.parse_args()

    if args.autoprueba or not args.alinear:
        return autoprueba()
    if not args.telefono:
        raise SystemExit("Falta --telefono con el CSV de Phyphox.")
    return alinea(REPO / args.alinear, Path(args.telefono), args.columnas,
                  Path(args.salida) if args.salida else None)


if __name__ == "__main__":
    sys.exit(main())
