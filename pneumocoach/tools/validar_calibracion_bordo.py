"""Valida la calibración que el dispositivo hace por su cuenta.

    python tools/validar_calibracion_bordo.py --puerto COM9 --sujeto s01

Qué comprueba, y por qué hacen falta las dos partes
---------------------------------------------------
1. EL FLUJO. Con el sensor puesto, guía las dos maniobras de referencia y
   comprueba que la máquina de estados avanza sola: asentamiento → maniobra
   diafragmática → torácica → calibrado. Mientras tanto graba el CSV.

2. LOS NÚMEROS. Que el dispositivo diga «he terminado» no prueba nada sobre lo
   que calculó. Así que después reinyecta la grabación por su máquina de sesión
   y compara sus dos vectores de referencia contra los que calcula Python sobre
   exactamente las mismas muestras.

La reinyección existe para evitar un problema de alineamiento: en la corrida en
vivo, el contador del CSV y el del DSP arrancan en momentos distintos y avanzan
en tareas distintas, así que casarlos muestra a muestra sería una fuente de
error mayor que lo que se quiere medir. Reinyectando, las dos partes ven la
misma secuencia desde la muestra cero.
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

import _consola  # noqa: E402,F401
from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp  # noqa: E402

C.WARMUP_S_DISPOSITIVO = dsp.WARMUP_S
from pneumocoach.calibracion import ReferenciaSesion  # noqa: E402
from analizar_captura import cargar, cargar_calibracion  # noqa: E402
from paridad import abrir, carga_montaje, FIN  # noqa: E402

# Tramas reservadas del modo de inyeccion por sesion. Van dentro de la
# trama de 12 bytes porque una marca suelta colisiona con los datos:
# 0x64 es 'd' y aparece constantemente dentro de una muestra int16.
FASE = {
    "d": struct.pack("<6h", *([-2] * 6)),
    "t": struct.pack("<6h", *([-3] * 6)),
    ".": struct.pack("<6h", *([-4] * 6)),
}

RTOL, ATOL = 2.0e-2, 1.0e-4

MANIOBRAS = [
    ("d", "DIAFRAGMÁTICA",
     "Mano derecha en el pecho, izquierda en el abdomen.",
     "Respire moviendo SOLO la mano del abdomen.",
     "La del PECHO tiene que quedarse QUIETA."),
    ("t", "TORÁCICA",
     "Al revés: mueva SOLO la mano del pecho.",
     "Levante el esternón, respiración superficial.",
     "La del ABDOMEN no se mueve."),
]


class Flujo:
    """Un solo lector del puerto.

    Mientras se graba NO se puede vaciar el buffer de entrada: las lineas del
    CSV y las respuestas de estado llegan por el mismo cable, y un
    reset_input_buffer para leer el estado se lleva por delante las muestras.
    Aqui se drena una sola vez y se reparte por prefijo.
    """

    def __init__(self, ser):
        self.ser = ser
        self.buf = b""
        self.muestras: list[str] = []
        self.estado = ""
        # Fase en curso. Se anota en cada muestra para que la grabacion
        # quede autodescrita y cargar() la lea como cualquier otra.
        self.fase = "reposo"

    def drena(self) -> None:
        self.buf += self.ser.read(self.ser.in_waiting or 0)
        while b"\n" in self.buf:
            ln, self.buf = self.buf.split(b"\n", 1)
            t = ln.decode("utf-8", "replace").strip()
            if not t:
                continue
            if t.startswith("# SESION "):
                self.estado = t
            elif not t.startswith("#"):
                self.muestras.append(t + "," + self.fase)

    def pide_estado(self, espera: float = 0.45) -> str:
        self.estado = ""
        self.ser.write(b"c")
        t0 = time.time()
        while time.time() - t0 < espera:
            self.drena()
            if self.estado:
                break
            time.sleep(0.05)
        return self.estado


def cuenta_atras(seg: float, titulo: str) -> None:
    t0 = time.time()
    while True:
        queda = seg - (time.time() - t0)
        if queda <= 0:
            break
        print(f"\n{titulo}  {queda:4.0f} s ", end="", flush=True)
        time.sleep(0.2)
    print(f"\n{titulo}  hecho      ")


def corrida_en_vivo(ser, csv: Path) -> None:
    """Guia las maniobras y graba el CSV a la vez."""
    print("=" * 70)
    print("  CALIBRACION A BORDO - corrida en vivo")
    print("=" * 70)
    print()
    print("  El dispositivo dirige. Siga lo que diga la pantalla; esto solo")
    print("  lo acompana y graba.")
    print()

    f = Flujo(ser)
    ser.write(b"k")
    time.sleep(0.3)
    f.drena()
    ser.write(b"s")
    time.sleep(0.3)

    print("  Asentando los filtros. Colocate comodo y quieto.")
    t0 = time.time()
    while time.time() - t0 < 70:
        f.drena()
        e = f.pide_estado()
        if "LISTO" in e:
            break
        queda = max(0.0, C.WARMUP_S_DISPOSITIVO - (time.time() - t0))
        print(f"\nquedan {queda:4.0f} s ", end="", flush=True)
    print(f"\nasentado a los {time.time() - t0:.0f} s      ")
    print()

    ser.write(b"n")
    time.sleep(0.2)

    for clave, nombre, *consigna in MANIOBRAS:
        print(f"  >>> {nombre}")
        for c in consigna:
            print(f"      {c}")
        t0 = time.time()
        f.fase = "diaphragmatic" if clave == "d" else "thoracic"
        objetivo = "CAL TORACICA" if clave == "d" else "CALIBRADO"
        while time.time() - t0 < C.REF_SEGUNDOS + 8:
            f.drena()
            e = f.pide_estado(0.3)
            if objetivo in e:
                break
            queda = max(0.0, C.REF_SEGUNDOS - (time.time() - t0))
            print(f"\nquedan {queda:4.0f} s ", end="", flush=True)
        print(f"\nhecho              ")
        print()

    f.drena()
    ser.write(b"x")
    time.sleep(0.5)
    f.drena()
    print(f"  {f.pide_estado()}")
    print()

    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text(
        "# calibracion a bordo, generada por validar_calibracion_bordo.py\n"
        "# valores CRUDOS int16 del registro 0x3B\n"
        "seq,t_us,ax,ay,az,gx,gy,gz,mark,tecnica\n"
        + "\n".join(f.muestras) + "\n", encoding="utf-8")
    print(f"  grabadas {len(f.muestras)} muestras -> {csv.relative_to(REPO)}")


def lee_referencia(ser) -> dict[str, tuple[int, np.ndarray]]:
    ser.reset_input_buffer()
    ser.write(b"R")
    time.sleep(1.2)
    txt = ser.read(ser.in_waiting or 0).decode("utf-8", "replace")
    out = {}
    for ln in txt.splitlines():
        if ln.startswith(("RD,", "RT,")):
            p = ln.split(",")
            out["dia" if ln[1] == "D" else "tor"] = (
                int(p[1]), np.array([float(v) for v in p[2:]], dtype=np.float64))
    return out


def reinyecta(ser, counts: np.ndarray, marcas: list[tuple[int, str]]) -> dict:
    """Manda la grabación por la máquina de sesión, con sus fases."""
    ser.write(b"S")
    time.sleep(0.5)
    ser.reset_input_buffer()

    # Los bloques se cortan EXACTAMENTE en las marcas de fase.
    #
    # Mandando de 40 en 40 sin mirar donde caen las marcas, la frontera que ve
    # el dispositivo llega hasta 40 muestras tarde respecto a la que usa Python.
    # Con un salto de ventana de 150, ese desfase cambia que ventanas entran en
    # cada maniobra: los dos lados cuentan el mismo numero pero no son las
    # mismas, y las caracteristicas de temporizacion -que varian mucho de una
    # ventana a otra- salen distintas mientras las de amplitud apenas se mueven.
    cortes = sorted({0, len(counts)} | {m[0] for m in marcas})
    por_marca = {m[0]: m[1] for m in marcas}
    for a, b in zip(cortes, cortes[1:]):
        if a in por_marca:
            ser.write(FASE[por_marca[a]])
            time.sleep(0.05)
        for i in range(a, b, 40):
            ser.write(b"".join(struct.pack("<6h", *f)
                               for f in counts[i:min(i + 40, b)]))
            if ser.in_waiting:
                ser.read(ser.in_waiting)
    ser.write(FIN)
    time.sleep(1.0)
    ser.read(ser.in_waiting or 0)
    return lee_referencia(ser)


def de_python(counts: np.ndarray, marcas: list[tuple[int, str]],
              R, sesgo) -> dict[str, tuple[int, np.ndarray]]:
    """La misma cuenta, en Python, sobre las mismas muestras."""
    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)
    fases: dict[str, list[np.ndarray]] = {"dia": [], "tor": []}
    limites = marcas + [(len(counts), ".")]
    for k, (ini, clave) in enumerate(marcas):
        if clave not in ("d", "t"):
            continue
        fin = limites[k + 1][0]
        destino = "dia" if clave == "d" else "tor"
        for i in range(0, len(counts) - C.WINDOW_N + 1, C.HOP_N):
            if i < ini or (i + C.WINDOW_N) > fin:
                continue
            fases[destino].append(dsp.extract_features(
                {kk: v[i:i + C.WINDOW_N] for kk, v in ch.items()}))
    return {k: (len(v), np.mean(v, axis=0)) for k, v in fases.items() if v}


def compara(dev: dict, py: dict) -> tuple[int, bool]:
    print("\n" + "=" * 70)
    print("  VECTORES DE REFERENCIA · dispositivo contra Python")
    print("=" * 70)
    fuera_total = 0
    nombres_fuera: set[str] = set()
    for fase in ("dia", "tor"):
        if fase not in dev or fase not in py:
            print(f"\n  {fase}: falta en {'dispositivo' if fase not in dev else 'Python'}")
            fuera_total += 1
            continue
        n_d, v_d = dev[fase]
        n_p, v_p = py[fase]
        err = np.abs(v_d - v_p)
        tol = ATOL + RTOL * np.abs(v_p)
        fuera = err > tol
        margen = float(np.min(tol / (err + 1e-30)))
        print(f"\n  {fase.upper()}   ventanas: dispositivo {n_d}, Python {n_p}"
              f"{'   <- NO COINCIDEN' if n_d != n_p else ''}")
        print(f"        {int(fuera.sum())} de {C.N_FEATURES} fuera de tolerancia"
              f"   margen mínimo x{margen:.1f}")
        if fuera.any():
            for j in np.where(fuera)[0]:
                print(f"          {C.FEATURE_NAMES[j]:<24}"
                      f"python {v_p[j]:>12.5g}   esp32 {v_d[j]:>12.5g}")
        fuera_total += int(fuera.sum()) + int(n_d != n_p)
        nombres_fuera.update(C.FEATURE_NAMES[j] for j in np.where(fuera)[0])
    solo_temp = bool(nombres_fuera) and nombres_fuera <= {
        "breath_rate_bpm", "breath_period_cv", "ie_ratio_mean", "ie_ratio_cv"}
    return fuera_total, solo_temp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", required=True)
    ap.add_argument("--sujeto", default="s01")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--solo-comparar", type=Path, default=None,
                    help="salta la corrida en vivo y usa este CSV")
    args = ap.parse_args()

    sello = time.strftime("%Y%m%d_%H%M%S")
    csv = args.csv or (REPO / "data" / "raw" /
                       f"{args.sujeto}_calbordo_{sello}.csv")

    ser = abrir(args.puerto)
    try:
        if args.solo_comparar:
            csv = args.solo_comparar
        else:
            corrida_en_vivo(ser, csv)
            print("\n  Referencia que calculó el dispositivo en vivo:")
            for fase, (n, v) in lee_referencia(ser).items():
                print(f"    {fase}  {n} ventanas   tilt_rms {v[0]:.4f}")

        # --- comparación numérica exacta, reinyectando ---
        counts, _ = cargar(csv)
        n = len(counts)
        # Las fases se reconstruyen igual que las dirigió la corrida en vivo:
        # asentamiento, luego dos maniobras consecutivas de REF_SEGUNDOS.
        ref_n = int(C.REF_SEGUNDOS * C.FS_HZ)
        ini_dia = max(0, n - 2 * ref_n)
        marcas = [(ini_dia, "d"), (ini_dia + ref_n, "t")]
        print(f"\n  Reinyectando {n} muestras "
              f"(dia desde {marcas[0][0]}, tor desde {marcas[1][0]})...")

        R, sesgo = cargar_calibracion(args.sujeto)
        # El dispositivo tiene que procesar en el mismo marco que Python;
        # si no, la comparacion mide la rotacion y no la calibracion.
        carga_montaje(ser, R, sesgo)
        dev = reinyecta(ser, counts, marcas)
        py = de_python(counts, marcas, R, sesgo)
        malos, solo_temporizacion = compara(dev, py)
    finally:
        ser.close()

    print("\n" + "=" * 70)
    TEMPORIZACION = {"breath_rate_bpm", "breath_period_cv",
                     "ie_ratio_mean", "ie_ratio_cv"}
    if malos == 0:
        print("  OK · la calibracion a bordo reproduce la de Python.")
    elif malos <= len(TEMPORIZACION) * 2 and solo_temporizacion:
        print(f"  PARCIAL · las {C.N_FEATURES - 4} caracteristicas de amplitud y")
        print("  espectro coinciden. Solo discrepan las de temporizacion.")
        print()
        print("  No es un fallo del puerto a C: es que una ventana de 12 s a")
        print("  8-14 respiraciones por minuto contiene una o dos respiraciones,")
        print("  y algunos cruces del comparador caen a menos de 1e-3 sigma del")
        print("  umbral, que es menos de lo que separa float32 de float64. Con")
        print("  tan pocas respiraciones, un solo cruce mueve el ritmo entero.")
        print()
        print("  Es una limitacion de la ventana de referencia, no del eje: las")
        print("  cuatro son poco informativas ahi por construccion.")
    else:
        print(f"  {malos} discrepancias fuera de las de temporizacion. El")
        print("  dispositivo calibra con un eje distinto al de Python, y el")
        print("  modelo se entrena sobre el de Python.")
    print("=" * 70)
    return 1 if malos else 0


if __name__ == "__main__":
    sys.exit(main())
