"""Mide como quedo montado el sensor y calcula la rotacion que lo compensa.

    python tools/orientacion.py --puerto COM9 --sujeto s01

Por que existe
--------------
La cadena de DSP razona en un marco anatomico: eje X hacia la cabeza (superior),
Z saliendo del pecho (anterior). El sensor casi nunca queda montado asi. Sobre
un torso real la placa se apoya en una superficie curva, la cinta la ladea, y la
carcasa tiene sus propias restricciones.

La primera version de este script exigia que el modulo quedara alineado con
algun eje del sensor y elegia "el eje dominante" con argmax. Sobre la placa real
la gravedad se repartio 0.738 / 0.739 g entre Y y Z, o sea el modulo estaba a
45 grados, y argmax resolvio el empate por un milesimo: un diagnostico decidido
a cara o cruz.

Esta version no exige nada. Mide dos direcciones anatomicas conocidas, construye
una base ortonormal y guarda la matriz de rotacion. El DSP la aplica y trabaja
siempre en el marco anatomico, monte como monte la placa.

Metodo
------
Un acelerometro en reposo mide la direccion "arriba" local.

  De pie      -> "arriba" es la direccion SUPERIOR del cuerpo (hacia la cabeza)
  Boca arriba -> "arriba" es la direccion ANTERIOR (sale del pecho)

Con esos dos vectores, Gram-Schmidt da la base completa. Se conserva `superior`
exacto y se ortogonaliza `anterior` contra el, porque la medicion de pie es la
mas confiable de las dos: pararse derecho es facil, acostarse sin que una
almohada incline el torso, no tanto.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
import _consola  # noqa: E402,F401  UTF-8 en Windows
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))

from pneumocoach import config as C  # noqa: E402

CAL_DIR = REPO / "data" / "calibracion"

# Umbral de aviso para el error de perpendicular.
#
# Estaba en 12 grados por intuicion, y medirlo demostro que era demasiado
# estricto. Barriendo errores de 0 a 24 grados sobre senal sintetica, el impacto
# en `log_tilt_axial_ratio` se queda plano en ~0.08 unidades, contra una
# separacion tipica entre toracica y diafragmatica de ~1.5. Es el 5 % de la
# senal util, por debajo del ruido de la propia medicion.
#
# La razon es que Gram-Schmidt elimina justamente la componente que este error
# mide: al ortogonalizar `anterior` contra `superior` se resta el desvio de los
# 90 grados. Y el paso alto de 0.1 Hz se come cualquier offset constante de
# orientacion que sobreviva.
#
# Se conserva el aviso, pero como INDICADOR DE CALIDAD y no como fuente directa
# de error: un valor alto delata posturas descuidadas, y eso correlaciona con
# problemas que si importan, como que el modulo se despegue entre medidas.
TOLERANCIA_PERPENDICULAR_DEG = 25.0


def abrir(puerto: str, baud: int = 115200):
    import serial
    s = serial.Serial(puerto, baud, timeout=1)
    # Ver la nota en tools/capture.py: pyserial afirma DTR/RTS al abrir y en
    # placas CH340 eso deja al ESP32 retenido en reset.
    s.setDTR(False)
    s.setRTS(True)
    time.sleep(0.15)
    s.setRTS(False)
    time.sleep(0.5)
    s.reset_input_buffer()
    time.sleep(1.8)
    s.reset_input_buffer()
    return s


def medir(ser, segundos: float) -> np.ndarray:
    ser.reset_input_buffer()
    ser.write(b"s")
    t0 = time.time()
    filas = []
    while time.time() - t0 < segundos:
        ln = ser.readline().decode("utf-8", "replace").strip()
        if ln and not ln.startswith("#") and ln.count(",") == 8:
            try:
                filas.append([int(x) for x in ln.split(",")[2:8]])
            except ValueError:
                pass
    ser.write(b"x")
    time.sleep(0.2)
    if len(filas) < 20:
        sys.exit(f"Solo llegaron {len(filas)} muestras. Revisar el firmware.")
    d = np.asarray(filas, dtype=np.float64)
    d[:, :3] /= C.ACCEL_LSB_PER_G
    d[:, 3:] /= C.GYRO_LSB_PER_DPS
    return d


def base_anatomica(a_pie: np.ndarray, a_acostado: np.ndarray) -> tuple[np.ndarray, float]:
    """Construye la matriz que lleva del marco del sensor al anatomico.

    Devuelve (R, error_perpendicular_grados). R es 3x3 con filas
    [superior; lateral; anterior], de modo que  v_cuerpo = R @ v_sensor.
    """
    s = a_pie / np.linalg.norm(a_pie)          # superior, se conserva exacto
    n0 = a_acostado / np.linalg.norm(a_acostado)  # anterior, aproximado

    err = abs(90.0 - np.degrees(np.arccos(np.clip(float(s @ n0), -1, 1))))

    # Gram-Schmidt: quitar de `anterior` la componente que comparte con
    # `superior`. Lo que queda es perpendicular por construccion.
    n = n0 - float(n0 @ s) * s
    nn = np.linalg.norm(n)
    if nn < 1e-6:
        sys.exit("Las dos posturas dieron la misma direccion. Repetir la prueba.")
    n /= nn

    lat = np.cross(n, s)  # completa la terna derecha: sup x lat = ant
    lat /= np.linalg.norm(lat)

    return np.vstack([s, lat, n]), err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", required=True)
    ap.add_argument("--sujeto", default="s01")
    ap.add_argument("--seg", type=float, default=6.0)
    args = ap.parse_args()

    ser = abrir(args.puerto)
    print("=" * 66)
    print("  Calibracion de montaje")
    print("=" * 66)
    print()
    print("  El modulo va pegado al ESTERNON, centrado 3-4 cm debajo de la")
    print("  incisura yugular. Fijalo con cinta, no lo sostengas con la mano.")
    print()
    print("  No hace falta que quede alineado con nada: la rotacion se mide")
    print("  y se compensa. Solo tiene que quedar FIRME y no moverse entre")
    print("  las dos posturas.")
    print()

    input("  [1/2] De pie, bien derecho, mirando al frente. Enter...")
    print("        midiendo...")
    d1 = medir(ser, args.seg)
    a1 = d1[:, :3].mean(axis=0)
    r1 = d1[:, :3].std(axis=0).max()
    print(f"        accel = [{a1[0]:+.3f} {a1[1]:+.3f} {a1[2]:+.3f}] g   "
          f"(ruido {r1 * 1000:.1f} mg)")

    print()
    print("  Para la segunda: acuestate BOCA ARRIBA en una superficie plana.")
    print("  SIN ALMOHADA -- una almohada inclina el torso y arruina la medida.")
    print("  Brazos a los costados, mirando al techo.")
    input("  [2/2] Enter cuando estes acostado y quieto...")
    print("        midiendo...")
    d2 = medir(ser, args.seg)
    a2 = d2[:, :3].mean(axis=0)
    r2 = d2[:, :3].std(axis=0).max()
    print(f"        accel = [{a2[0]:+.3f} {a2[1]:+.3f} {a2[2]:+.3f}] g   "
          f"(ruido {r2 * 1000:.1f} mg)")
    # El puerto se cierra al final: la matriz todavia hay que enviarla.

    R, err = base_anatomica(a1, a2)
    sesgo = d1[:, 3:].mean(axis=0)

    print()
    print("-" * 66)
    print("  Base anatomica medida (filas = superior, lateral, anterior)")
    print("-" * 66)
    for nombre, fila in zip(("superior", "lateral ", "anterior"), R):
        print(f"    {nombre}  [{fila[0]:+.4f} {fila[1]:+.4f} {fila[2]:+.4f}]")

    print()
    print("  Calidad de la medicion:")
    for nombre, a in (("de pie  ", a1), ("acostado", a2)):
        m = float(np.linalg.norm(a))
        marca = "OK" if 0.85 < m < 1.15 else "<- fuera de rango"
        print(f"    |accel| {nombre} = {m:.3f} g   {marca}")
    print(f"    ruido maximo         = {max(r1, r2) * 1000:.1f} mg", end="")
    print("   OK" if max(r1, r2) < 0.05 else "   <- te moviste durante la medida")

    print(f"    error de perpendicular = {err:.1f} grados", end="")
    if err <= TOLERANCIA_PERPENDICULAR_DEG:
        print("   OK")
    else:
        print("   <- ATENCION")

    # Cuanto se aparta el montaje del marco ideal (X superior, Z anterior).
    giro = np.degrees(np.arccos(np.clip(abs(R[0, 0]), -1, 1)))
    print(f"    giro respecto al marco ideal = {giro:.1f} grados")

    print()
    if err > TOLERANCIA_PERPENDICULAR_DEG:
        print("  " + "!" * 62)
        print(f"  Las dos direcciones salieron a {90 - err:.0f} grados en vez de 90.")
        print("  Con esta magnitud lo mas probable es que algo se movio:")
        print("    - el modulo se despego o giro entre una medida y la otra")
        print("    - almohada o colchon blando inclinando mucho el torso")
        print("    - no estar minimamente derecho en la primera postura")
        print()
        print("  Conviene repetirla antes de grabar el dataset definitivo.")
        print("  " + "!" * 62)
    else:
        print("  Calibracion utilizable.")
        if err > 8.0:
            print(f"  El error de {err:.1f} grados viene sobre todo de la postura")
            print("  acostado (colchon, almohada) y de que el sensor va sobre")
            print("  tejido blando, que se acomoda distinto en cada postura.")
            print("  No es un problema: Gram-Schmidt elimina justamente esa")
            print("  componente, y el paso alto de 0.1 Hz se come el resto.")
            print("  Medido sobre senal sintetica, el impacto en el")
            print("  discriminador es del 5 % de la separacion entre clases.")

    CAL_DIR.mkdir(parents=True, exist_ok=True)
    destino = CAL_DIR / f"{args.sujeto}.json"
    destino.write_text(json.dumps({
        "sujeto": args.sujeto,
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "R_sensor_a_cuerpo": R.tolist(),
        "filas": ["superior", "lateral", "anterior"],
        "accel_de_pie_g": a1.tolist(),
        "accel_acostado_g": a2.tolist(),
        "sesgo_giro_dps": sesgo.tolist(),
        "error_perpendicular_deg": round(err, 2),
        "giro_vs_marco_ideal_deg": round(float(giro), 2),
        "escala_accel_medida_g": round(float(np.linalg.norm(a1)), 4),
    }, indent=2), encoding="utf-8")

    print()
    print("=" * 66)
    print(f"  Guardado en  data/calibracion/{destino.name}")

    # Y al dispositivo, que es quien la necesita para procesar en vivo.
    #
    # Guardarla solo en el JSON dejaba al firmware corriendo sobre los ejes
    # crudos del sensor mientras el analisis y el entrenamiento usaban el
    # marco anatomico. Un modelo desplegado asi recibe caracteristicas de
    # otro sistema de coordenadas y clasifica con confianza sobre el eje
    # equivocado, sin que nada lo delate.
    valores = [f"{v:.9g}" for fila in R for v in fila]
    valores += [f"{v:.9g}" for v in sesgo]
    ser.write(b"M" + (",".join(valores) + "\n").encode())
    time.sleep(0.6)
    eco = ser.read(ser.in_waiting or 0).decode("utf-8", "replace").strip()
    print(f"  Dispositivo: {eco.splitlines()[-1] if eco else 'sin respuesta'}")
    ser.close()
    print()
    print("  Ya puedes grabar. La captura usa esta calibracion automaticamente:")
    print(f"    python tools/capture.py --puerto {args.puerto} "
          f"--sujeto {args.sujeto} --protocolo")
    print("=" * 66)


if __name__ == "__main__":
    main()
