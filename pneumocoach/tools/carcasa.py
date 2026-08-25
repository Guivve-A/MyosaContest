"""Genera la carcasa del wearable: cotas, STL para imprimir y renders.

    python tools/carcasa.py cotas
    python tools/carcasa.py cotas --Wc 33.4 --Hc 32.1 --Tc 12.8
    python tools/carcasa.py stl --concepto ambos
    python tools/carcasa.py render --vistas tres_cuartos frontal

La geometria es parametrica. Los tres unicos numeros que hay que medir con
calibre sobre la placa fisica son `Wc`, `Hc` y `Tc` -el footprint y la altura
del carrier MYOSA con su breakout-; las cinco piezas se derivan de ellos. Eso
importa porque el carrier puede cambiar de lote: cuando cambie, se remide y se
regenera, en vez de rehacer el modelo.

Piezas
------
    C2   capsula esternal, de masa minima
    C3   base adhesiva con acoplamiento cinematico magnetico

La capsula se desacopla para cargar y vuelve al mismo asiento y la misma
orientacion. Eso no es comodidad: la colocacion forma parte de la medida. Sobre
el manubrio y sobre el apendice xifoides la mecanica es distinta y hasta de
signo contrario, asi que repetir la posicion es un requisito tecnico.

`cotas` no abre Blender
-----------------------
Comprobar si un carrier cabe en la zona util del esternon es aritmetica, y
tarda milisegundos. Renderizar tarda minutos. Separarlos permite descartar una
medida antes de gastar el tiempo.

Blender
-------
Se invoca en modo headless: sin interfaz, sin addons y sin que nadie tenga que
tener nada abierto. El resultado es determinista y funciona igual en un portatil
que en integracion continua. Se busca en las rutas habituales; para una
instalacion en otro sitio, definir la variable de entorno `BLENDER_EXE`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent
sys.path.insert(0, str(AQUI))

import _consola  # noqa: E402,F401

MODELO = AQUI / "carcasa_blender.py"
SALIDA = REPO / "data" / "mockups"

VISTAS = ("tres_cuartos", "frontal", "superior", "posterior")
CONCEPTOS = ("C2", "C3", "ambos")

#: Zona plana util del manubrio, en mm, y masa maxima del conjunto.
#: Salen del brief de diseno mecanico, no de una estimacion.
ZONA_W, ZONA_H, MASA_MAX = 40.0, 45.0, 25.0


def blender_exe() -> str:
    """Localiza Blender. La variable de entorno gana sobre la busqueda."""
    if env := os.environ.get("BLENDER_EXE"):
        if Path(env).exists():
            return env
    candidatos = [
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
    ]
    for c in candidatos:
        if Path(c).exists():
            return c
    base = Path(r"C:\Program Files\Blender Foundation")
    if base.exists():
        for exe in sorted(base.glob("*/blender.exe"), reverse=True):
            return str(exe)
    raise SystemExit(
        "No se encontro Blender. Instalalo, o define BLENDER_EXE con la ruta "
        "al ejecutable.")


def corre_blender(payload: dict[str, Any], timeout: int = 900) -> dict[str, Any]:
    """Invoca el script de modelo dentro de Blender y recoge lo que reporto."""
    SALIDA.mkdir(parents=True, exist_ok=True)
    payload.setdefault("outdir", str(SALIDA))

    cmd = [blender_exe(), "--background", "--factory-startup",
           "--python", str(MODELO), "--", json.dumps(payload)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    renders, stls, piezas, params = [], [], [], {}
    for ln in r.stdout.splitlines():
        if ln.startswith("RENDER "):
            renders.append(ln[7:].strip())
        elif ln.startswith("STL "):
            stls.append(ln[4:].strip())
        elif ln.startswith("PIEZA "):
            piezas.append(ln[6:].strip())
        elif ln.startswith("PARAMS "):
            params = json.loads(ln[7:])

    if not (renders or stls):
        # Sin artefactos: el final de stderr es donde Blender deja los
        # TypeError de la API de bpy cuando cambia entre versiones.
        cola = "\n".join(r.stderr.splitlines()[-25:]) or "sin salida"
        raise SystemExit(f"Blender no genero nada.\n\n{cola}")
    return {"renders": renders, "stl": stls, "piezas": piezas,
            "parametros": params}


def calcula_cotas(Wc: float, Hc: float, Tc: float,
                  pared: float = 1.6) -> dict[str, Any]:
    """Cotas exteriores derivadas y si el conjunto cabe. Sin abrir Blender."""
    ext_w, ext_h, ext_d = Wc + 5.0, Hc + 5.0, Tc + 3.5

    # Masa aproximada de un cascaron hueco. PETG son 1.27 g/cm3, y con 25 % de
    # relleno el efectivo ronda 0.45.
    v_ext = ext_w * ext_h * ext_d / 1000.0
    v_int = max(0.0, (Wc + 0.4) * (Hc + 0.4) * (Tc + 1.2) / 1000.0)
    masa = (v_ext - v_int) * 0.45

    avisos = []
    if ext_w > ZONA_W:
        avisos.append(
            f"El ancho exterior ({ext_w:.1f} mm) excede la zona plana del "
            f"manubrio ({ZONA_W:.0f} mm). Se apoyaria sobre las articulaciones "
            "costo-esternales, que se mueven distinto al esternon.")
    if ext_h > ZONA_H:
        avisos.append(f"El alto exterior ({ext_h:.1f} mm) excede los "
                      f"{ZONA_H:.0f} mm utiles.")
    if masa > MASA_MAX:
        avisos.append(
            f"Masa estimada {masa:.0f} g, por encima del limite de "
            f"{MASA_MAX:.0f} g. Baja la frecuencia de resonancia del montaje "
            "hacia la banda respiratoria.")

    return {
        "capsula_exterior_mm": [round(ext_w, 2), round(ext_h, 2), round(ext_d, 2)],
        "cavidad_interior_mm": [round(Wc + 0.4, 2), round(Hc + 0.4, 2),
                                round(Tc + 1.2, 2)],
        "base_adhesiva_mm": [46.0, 46.0, 2.5],
        "masa_estimada_g": round(masa, 1),
        "zona_util_mm": [ZONA_W, ZONA_H],
        "cabe": not avisos,
        "avisos": avisos,
    }


def imprime_cotas(c: dict[str, Any]) -> None:
    print("=" * 66)
    print("  COTAS DERIVADAS")
    print("=" * 66)
    print()
    w, h, d = c["capsula_exterior_mm"]
    iw, ih, idd = c["cavidad_interior_mm"]
    bw, bh, bd = c["base_adhesiva_mm"]
    print(f"  capsula exterior : {w:6.2f} x {h:6.2f} x {d:6.2f} mm")
    print(f"  cavidad interior : {iw:6.2f} x {ih:6.2f} x {idd:6.2f} mm")
    print(f"  base adhesiva    : {bw:6.2f} x {bh:6.2f} x {bd:6.2f} mm")
    print(f"  masa estimada    : {c['masa_estimada_g']:6.1f} g")
    print(f"  zona util        : {c['zona_util_mm'][0]:6.1f} x "
          f"{c['zona_util_mm'][1]:6.1f} mm")
    print()
    if c["cabe"]:
        print("  CABE en la zona util del esternon.")
    else:
        print("  NO CABE:")
        for a in c["avisos"]:
            print(f"    - {a}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="orden", required=True)

    def comunes(p):
        p.add_argument("--Wc", type=float, default=32.0,
                       help="ancho del carrier MYOSA, mm (medir con calibre)")
        p.add_argument("--Hc", type=float, default=32.0,
                       help="alto del carrier MYOSA, mm")
        p.add_argument("--Tc", type=float, default=12.0,
                       help="altura del stack con breakout, mm")
        p.add_argument("--pared", type=float, default=1.6,
                       help="espesor de pared, mm")

    comunes(sub.add_parser("cotas", help="cotas derivadas, sin abrir Blender"))

    p_stl = sub.add_parser("stl", help="exporta STL en milimetros")
    comunes(p_stl)
    p_stl.add_argument("--concepto", choices=CONCEPTOS, default="ambos")

    p_r = sub.add_parser("render", help="renderiza uno o varios encuadres")
    comunes(p_r)
    p_r.add_argument("--concepto", choices=CONCEPTOS, default="ambos")
    p_r.add_argument("--vistas", nargs="+", choices=VISTAS,
                     default=["tres_cuartos"])
    p_r.add_argument("--resolucion", type=int, default=1500)
    p_r.add_argument("--muestras", type=int, default=90,
                     help="muestras de Cycles; mas es mas limpio y mas lento")
    p_r.add_argument("--con-torso", action="store_true",
                     help="anade la referencia anatomica")

    a = ap.parse_args()
    cotas = calcula_cotas(a.Wc, a.Hc, a.Tc, a.pared)

    if a.orden == "cotas":
        imprime_cotas(cotas)
        return 0 if cotas["cabe"] else 1

    # Antes de gastar minutos en Blender, decir si la pieza cabe siquiera.
    if not cotas["cabe"]:
        imprime_cotas(cotas)
        print("  Se genera igualmente, pero revisa los avisos de arriba.")
        print()

    base = {"Wc": a.Wc, "Hc": a.Hc, "Tc": a.Tc, "wall": a.pared,
            "concept": a.concepto}
    if a.orden == "stl":
        r = corre_blender({**base, "render": False, "stl": True})
    else:
        r = corre_blender({**base, "render": True, "stl": False,
                           "vistas": a.vistas, "res": a.resolucion,
                           "samples": a.muestras, "torso": a.con_torso})

    for pieza in r["piezas"]:
        print(f"  pieza  {pieza}")
    for f in r["stl"] + r["renders"]:
        print(f"  {Path(f).relative_to(REPO) if str(f).startswith(str(REPO)) else f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
