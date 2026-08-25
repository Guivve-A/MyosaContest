"""Paridad Python <-> C, medida sobre el dispositivo real.

    python tools/paridad.py data/raw/<captura>.csv --puerto COM9

Inyecta las muestras crudas de una grabacion en el ESP32 por serial, deja que
el firmware corra su propio DSP, y compara las 29 caracteristicas que devuelve
contra las que calcula Python sobre exactamente los mismos datos.

Por que en el dispositivo y no en el host
-----------------------------------------
El unico compilador de C disponible en esta maquina es el cruzado de Xtensa, y
resulta ser la mejor opcion: la prueba corre sobre la FPU de precision simple
del ESP32, que es donde tiene que coincidir. Un DSP que casa en el PC con
float64 y falla en el dispositivo con float32 no sirve de nada.

Que se compara
--------------
Todo el encadenamiento: filtro complementario, biquads Butterworth en cascada,
diezmado, FFT de 128 puntos, segmentacion de respiraciones y las 29
caracteristicas. Si alguna pieza difiere, la caracteristica afectada se sale de
tolerancia y el informe dice cual.

La tolerancia es relativa 2e-2 con piso absoluto 1e-4. NumPy calcula en float64
y el ESP32 en float32, asi que nunca coincidiran bit a bit; cualquier
desviacion mayor es una diferencia de algoritmo, no de redondeo.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))


import _consola  # noqa: E402,F401  UTF-8 en Windows
from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402
from analizar_captura import cargar  # noqa: E402

RTOL, ATOL = 2.0e-2, 1.0e-4

# Piso absoluto distinto para las caracteristicas cuyo valor pasa por cero.
#
# El criterio relativo no dice nada cuando el valor es casi nulo: una
# correlacion de 0.015 frente a 0.014 significa lo mismo -"no correlacionados"-
# pero da un 7 % de error relativo. La regla, aplicada igual a las tres, es
# 0.004 sigma de la dispersion real de la caracteristica: por debajo de eso la
# diferencia no puede cambiar ningun veredicto.
#
# Las sigmas salen de 475 ventanas de las tres grabaciones reales; se
# recalculan con tools/medir_dispersion.py si cambia el conjunto de datos.
ATOL_POR_CARACTERISTICA = {
    "tilt_axial_xcorr": 1.0e-3,   # sigma 0.269, acotada en [-1, 1]
    "ie_ratio_cv": 1.3e-3,        # sigma 0.311
    "breath_period_cv": 7.5e-4,   # sigma 0.185
}
FIN = struct.pack("<6h", -1, -1, -1, -1, -1, -1)


def abrir(puerto: str):
    import serial
    s = serial.Serial(puerto, 115200, timeout=2)
    s.setDTR(False)
    s.setRTS(True)
    time.sleep(0.15)
    s.setRTS(False)
    time.sleep(0.6)
    s.reset_input_buffer()
    time.sleep(1.6)
    s.reset_input_buffer()
    return s


def carga_montaje(ser, mount, sesgo) -> None:
    """Pone en el dispositivo la matriz que corresponde a ESTA grabacion.

    El dispositivo guarda una sola matriz en NVS, y no tiene por que ser la de
    la grabacion que se va a inyectar. Si no coincide, los dos lados procesan en
    marcos distintos y la prueba falla por un motivo que no tiene nada que ver
    con el DSP. Cargarla aqui hace la prueba autocontenida.
    """
    if mount is None:
        return
    vals = [f"{v:.9g}" for fila in mount for v in fila]
    vals += [f"{v:.9g}" for v in (sesgo if sesgo is not None else [0, 0, 0])]
    ser.write(b"M" + (",".join(vals) + "\n").encode())
    time.sleep(0.8)
    eco = ser.read(ser.in_waiting or 0).decode("utf-8", "replace")
    print(f"    montaje cargado en el dispositivo: "
          f"{'ok' if 'guardado' in eco else 'SIN CONFIRMAR'}")


def del_dispositivo(ser, counts: np.ndarray) -> dict[int, np.ndarray]:
    """Envia las muestras y recoge {indice_inicio: vector de 29}."""
    ser.write(b"T")
    time.sleep(0.4)
    ser.reset_input_buffer()

    salida: dict[int, np.ndarray] = {}
    tiempos: list[float] = []
    buf = b""
    enviadas = 0
    total = len(counts)

    def drenar():
        nonlocal buf
        buf += ser.read(ser.in_waiting or 0)
        while b"\n" in buf:
            linea, buf = buf.split(b"\n", 1)
            txt = linea.decode("utf-8", "replace").strip()
            if not txt.startswith("F,"):
                continue
            p = txt.split(",")
            if len(p) != 2 + 1 + C.N_FEATURES:
                continue
            salida[int(p[1])] = np.array([float(v) for v in p[3:]], dtype=np.float64)
            tiempos.append(float(p[2]))

    # Enviar por bloques para no desbordar el buffer serial del dispositivo.
    BLOQUE = 40
    while enviadas < total:
        trozo = counts[enviadas:enviadas + BLOQUE]
        ser.write(b"".join(struct.pack("<6h", *fila) for fila in trozo))
        enviadas += len(trozo)
        drenar()
        if enviadas % 2000 == 0:
            print(f"\r    enviadas {enviadas}/{total}", end="", flush=True)
    print(f"\r    enviadas {total}/{total}          ")

    ser.write(FIN)
    t0 = time.time()
    while time.time() - t0 < 4:
        drenar()
        time.sleep(0.05)
    if tiempos:
        print(f"    extraccion en el dispositivo: {np.mean(tiempos)/1000:.2f} ms "
              f"por ventana (max {max(tiempos)/1000:.2f})")
    return salida


def de_python(counts: np.ndarray, mount=None, sesgo=None) -> dict[int, np.ndarray]:
    """Referencia de Python.

    El montaje entra aqui a proposito. Sin el, esta prueba comparaba los dos
    lados corriendo ambos sobre los ejes crudos del sensor: coincidian, pero
    ninguno de los dos hacia lo que hace el analisis de verdad, y el hecho de
    que el firmware nunca cargara la matriz paso desapercibido.
    """
    ch = dsp.channels_from_counts(counts, mount=mount, gyro_bias_dps=sesgo)
    out = {}
    for ini in range(0, len(counts) - C.WINDOW_N + 1, C.HOP_N):
        w = {k: v[ini:ini + C.WINDOW_N] for k, v in ch.items()}
        out[ini] = dsp.extract_features(w).astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--puerto", required=True)
    ap.add_argument("--sujeto", default="s01",
                    help="de quien es la matriz de montaje a usar")
    ap.add_argument("--guardar", type=Path, default=None,
                    help="npz con las dos matrices, para analizar sin volver "
                         "a inyectar la grabacion entera")
    ap.add_argument("--muestras", type=int, default=6000,
                    help="cuantas muestras inyectar (6000 = 120 s)")
    args = ap.parse_args()

    from analizar_captura import cargar_calibracion
    import json as _json
    _m = args.csv.with_suffix('.json')
    _f = (_json.loads(_m.read_text(encoding='utf-8')).get('inicio_utc')
          if _m.exists() else None)
    mount, sesgo = cargar_calibracion(args.sujeto, _f)
    counts, _ = cargar(args.csv)
    counts = counts[: args.muestras]
    print("=" * 74)
    print(f"  PARIDAD  ·  {args.csv.name}  ·  {len(counts)} muestras")
    print("=" * 74)

    print("\n  Inyectando en el dispositivo...")
    ser = abrir(args.puerto)
    carga_montaje(ser, mount, sesgo)
    dev = del_dispositivo(ser, counts)
    ser.close()
    print(f"    el dispositivo devolvio {len(dev)} ventanas")

    print("\n  Calculando en Python...")
    py = de_python(counts, mount, sesgo)
    print(f"    Python calculo {len(py)} ventanas")

    comunes = sorted(set(dev) & set(py))
    # El paso alto de 0.1 Hz tarda en asentarse; Python descarta ese prefijo y
    # aqui se hace lo mismo para no comparar transitorios.
    comunes = [i for i in comunes if i >= dsp.WARMUP_N]
    if not comunes:
        sys.exit("Sin ventanas comparables tras el calentamiento. Enviar mas muestras.")
    print(f"    {len(comunes)} ventanas comparables tras el calentamiento\n")

    D = np.stack([dev[i] for i in comunes])
    P = np.stack([py[i] for i in comunes])
    err_abs = np.abs(D - P)
    atol = np.array([ATOL_POR_CARACTERISTICA.get(n, ATOL) for n in C.FEATURE_NAMES])
    tol = atol + RTOL * np.abs(P)
    fuera = err_abs > tol
    # Margen: cuanto sobra hasta la tolerancia. Un test que pasa con margen 1.1
    # esta a un decimal de fallar con la siguiente grabacion, y eso tiene que
    # verse en el informe, no descubrirse el dia de la demo.
    margen = tol / (err_abs + 1e-30)

    print("=" * 74)
    print(f"  {'caracteristica':<24}{'python':>12}{'esp32':>12}{'err rel':>11}"
          f"{'n/=0':>6}{'margen':>8}{'':>6}")
    print("=" * 74)
    n_mal = 0
    sin_ejercitar = []
    estrechas: list[tuple[str, float]] = []
    for j, nombre in enumerate(C.FEATURE_NAMES):
        # Una caracteristica que vale cero en TODAS las ventanas no esta
        # validada aunque coincida: las dos implementaciones podrian estar
        # fallando igual. Se marca aparte en vez de contarla como acierto.
        if np.all(np.abs(P[:, j]) < 1e-9) and np.all(np.abs(D[:, j]) < 1e-9):
            sin_ejercitar.append(nombre)
        rel = err_abs[:, j] / (np.abs(P[:, j]) + 1e-12)
        peor = int(np.argmax(err_abs[:, j] - tol[:, j]))
        ok = not fuera[:, j].any()
        if not ok:
            n_mal += 1
        # Cobertura: en cuantas ventanas la caracteristica tomo un valor no
        # trivial. Coincidir en cero no demuestra nada; hay que saber sobre
        # cuantos valores reales se comprobo.
        viva = np.abs(P[:, j]) > 1e-9
        # Multiplicar por la mascara no sirve: los ceros de las ventanas
        # muertas ganan el argmax frente a los margenes negativos de las que
        # pasan holgadamente. Hay que excluirlas con -inf.
        score = np.where(viva, err_abs[:, j] - tol[:, j], -np.inf)
        peor_vivo = int(np.argmax(score)) if viva.any() else peor
        m = float(margen[:, j].min())
        marca = "  ok" if ok else f"  FUERA ({fuera[:, j].sum()}/{len(comunes)})"
        if ok and m < 2.0:
            marca = "  ok (justo)"
            estrechas.append((nombre, m))
        print(f"  {nombre:<24}{P[peor_vivo, j]:>12.5g}{D[peor_vivo, j]:>12.5g}"
              f"{rel[peor_vivo]:>11.2e}{int(viva.sum()):>6}{m:>8.1f}{marca}")

    if estrechas:
        print()
        print("  Pasan, pero con poco margen. Si el margen baja de 1 el test")
        print("  falla; conviene entender por que antes de que ocurra solo:")
        for n, m in sorted(estrechas, key=lambda x: x[1]):
            print(f"    - {n}  margen x{m:.2f}")

    if sin_ejercitar:
        print()
        print(f"  {len(sin_ejercitar)} caracteristicas valen CERO en las dos")
        print("  implementaciones y en todas las ventanas. Coinciden, pero eso no")
        print("  las valida: podrian estar fallando igual. Hacen falta ventanas")
        print("  que las ejerciten.")
        for n in sin_ejercitar:
            print(f"    - {n}")

    print("=" * 74)
    if sin_ejercitar:
        print(f"  PARIDAD PARCIAL · {C.N_FEATURES - len(sin_ejercitar)} de "
              f"{C.N_FEATURES} caracteristicas validadas con valores no triviales.")
        print(f"  Las otras {len(sin_ejercitar)} no se ejercitaron en esta corrida.")
    elif n_mal == 0:
        print(f"  PARIDAD OK · las {C.N_FEATURES} caracteristicas coinciden dentro de")
        print(f"  rtol={RTOL:g} atol={ATOL:g} sobre {len(comunes)} ventanas.")
        print("  El DSP en C reproduce el de Python sobre hardware real.")
    else:
        print(f"  {n_mal} de {C.N_FEATURES} caracteristicas FUERA de tolerancia.")
        print("  El modelo se entreno sobre la salida de Python: cualquier")
        print("  caracteristica que no coincida produce veredictos equivocados.")
    if args.guardar:
        args.guardar.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.guardar, python=P, esp32=D, ventanas=np.array(comunes),
                 nombres=np.array(C.FEATURE_NAMES))
        print(f"  guardado en {args.guardar}")
    print("=" * 74)
    return 1 if n_mal else 0


if __name__ == "__main__":
    sys.exit(main())
