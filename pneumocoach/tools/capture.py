"""Captura sesiones etiquetadas desde el firmware de adquisicion.

    python tools/capture.py --puerto COM5 --sujeto s01 --tecnica diaphragmatic
    python tools/capture.py --listar-puertos
    python tools/capture.py --puerto COM5 --protocolo --sujeto s01

El etiquetado lo impone la herramienta, no la disciplina de quien graba. Las
etiquetas nacen del guion cronometrado que la propia herramienta dicta en
pantalla, se escriben en la cabecera del CSV y en un JSON aparte, y el nombre
del archivo se deriva de los metadatos. Nadie teclea "toracica_v2_final_final".

Ese detalle no es burocracia: la precision de estas etiquetas es el techo de
todo lo que el modelo pueda aprender despues.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
import _consola  # noqa: E402,F401  UTF-8 en Windows
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "raw"

# Debe coincidir con ml/pneumocoach/config.py
FS_HZ = 50.0
COLUMNS = ["seq", "t_us", "ax", "ay", "az", "gx", "gy", "gz", "mark"]

TECNICAS = {
    "diaphragmatic": "Diafragmatica: abdomen se expande, pecho superior quieto, "
                     "exhalacion larga con labios fruncidos",
    "thoracic": "Toracica: respirar solo con el pecho, superficial, "
                "hombros y esternon se levantan",
    "rapid_shallow": "Rapida superficial: 25-40 por minuto, poco volumen, "
                     "como despues de subir escaleras",
    "apnea": "Apnea voluntaria: contener la respiracion, sin moverse",
    "artifact": "Artefacto: toser, hablar, moverse en la silla, rascarse",
    "reposo": "Reposo: respirar normal sin pensarlo, linea base",
    "dia_suave": "Diafragmatica SUAVE: tecnica correcta, poco esfuerzo",
    "dia_fuerte": "Diafragmatica EXAGERADA: tecnica correcta, muy amplia",
    "tor_suave": "Toracica SUAVE: tecnica incorrecta, poco esfuerzo",
    "tor_fuerte": "Toracica EXAGERADA: tecnica incorrecta, muy amplia",
}

# Guion cronometrado de una sesion completa. Segundos por bloque.
# El orden alterna tecnicas para que la deriva postural no se correlacione con
# la etiqueta: si todas las diafragmaticas fueran al principio, el modelo
# podria aprender "el sujeto todavia no se ha acomodado" en vez de la tecnica.
# CONSIGNAS DE DOS MANOS.
#
# La primera version de este protocolo describia la tecnica sin decir que parte
# del cuerpo NO debia moverse, y el sujeto hizo lo previsible: al pedirle
# diafragmatica movio tambien el esternon. La etiqueta decia diafragmatica y la
# mecanica era toracica, y ese es exactamente el problema de etiquetado que
# documenta ADR-0007.
#
# La correccion es la senal manual de la fisioterapia respiratoria: una mano en
# el pecho y otra en el abdomen, y la consigna dice cual se mueve y cual NO.
# Restringir el esternon es parte de la definicion de la tecnica, no un matiz.
PROTOCOLO = [
    ("reposo", 60, "Sientese comodo. Respire normal, sin pensar en como."),
    ("diaphragmatic", 120,
     "Mano derecha en el pecho, izquierda en el abdomen. Inhale hinchando la "
     "panza y exhale LARGO con labios fruncidos, el doble de tiempo. "
     "La mano del PECHO tiene que quedarse QUIETA."),
    ("thoracic", 120,
     "Al reves: mueva SOLO la mano del pecho. Levante el esternon, "
     "respiracion superficial. La mano del ABDOMEN no se mueve."),
    ("apnea", 45, "Inhale normal y contenga. Avise si necesita parar."),
    ("rapid_shallow", 90, "Respire rapido y poco profundo, como jadeando suave."),
    ("diaphragmatic", 120,
     "Otra vez diafragmatica. Solo la mano del abdomen; la del pecho QUIETA."),
    ("artifact", 60, "Tosa 3 veces, hable 15 segundos, acomodese en la silla, "
                     "rasquese el hombro."),
    ("thoracic", 120,
     "Otra vez toracica. Solo la mano del pecho; la del abdomen QUIETA."),
    ("reposo", 60, "Vuelva a respirar normal. Fin de la sesion."),
]


# Protocolo de esfuerzo: separa TECNICA de INTENSIDAD.
#
# El primer sujeto real dejo un discriminador basado en amplitud absoluta, lo
# que abre una pregunta que invalida el enfoque si sale mal: puede el esfuerzo
# falsificar la tecnica? Si una toracica floja se ve igual que una
# diafragmatica exagerada, la amplitud no sirve ni normalizada.
#
# El ORDEN no es arbitrario. La fatiga hace que los bloques tardios tengan
# menos amplitud, y eso se confundiria con el efecto que medimos. Por eso los
# dos bloques de la comparacion critica -toracica suave contra diafragmatica
# exagerada- van ADYACENTES en el centro, donde comparten estado de fatiga.
# CONSIGNAS CORREGIDAS tras el primer intento.
#
# La version anterior ponia SUAVE/EXAGERADA en mayusculas y dejaba la tecnica
# como adjetivo. El sujeto siguio la instruccion saliente: al pedirle
# "diafragmatica exagerada" exagero la respiracion entera, esternon incluido, y
# produjo 6.86 grados de rotacion esternal -la mas alta de toda la sesion, casi
# el triple que su propia respiracion toracica-. La etiqueta decia
# diafragmatica y la mecanica era toracica.
#
# La correccion es la senal manual que usa la fisioterapia respiratoria desde
# siempre: una mano en el pecho y otra en el abdomen, y la consigna dice
# explicitamente cual debe moverse y cual NO. Restringir el esternon es parte de
# la definicion de la tecnica, no un detalle.
PROTOCOLO_ESFUERZO = [
    ("reposo", 60, "Respire normal, sin pensar. Linea base."),
    ("dia_suave", 60,
     "Mano derecha en el pecho, izquierda en el abdomen. Respire moviendo "
     "SOLO la mano del abdomen, y poco. La del pecho NO se mueve."),
    ("tor_fuerte", 60,
     "Ahora al reves: mueva SOLO la mano del pecho, y mucho. Levante "
     "esternon y hombros. La del abdomen NO se mueve."),
    ("tor_suave", 60,
     "Siga moviendo solo la mano del pecho, pero apenas. Movimiento minimo. "
     "La del abdomen sigue quieta."),
    ("dia_fuerte", 60,
     "Vuelva al abdomen: hinche la panza AL MAXIMO, exhale muy largo con "
     "labios fruncidos. La mano del pecho tiene que seguir QUIETA. Si el "
     "esternon se levanta, no es diafragmatica."),
    ("reposo", 60, "Vuelva a respirar normal. Fin."),
]


@dataclass
class Bloque:
    tecnica: str
    inicio_s: float
    fin_s: float
    consigna: str


@dataclass
class Sesion:
    sujeto: str
    fs_hz: float
    inicio_utc: str
    colocacion: str
    postura: str
    notas: str
    operador: str
    bloques: list[Bloque] = field(default_factory=list)
    n_muestras: int = 0
    n_perdidas: int = 0
    duracion_s: float = 0.0


def listar_puertos() -> None:
    try:
        from serial.tools import list_ports
    except ImportError:
        sys.exit("Falta pyserial.  pip install pyserial")
    puertos = list(list_ports.comports())
    if not puertos:
        print("No se detectaron puertos serie.")
        return
    print("Puertos disponibles:")
    for p in puertos:
        print(f"  {p.device:10} {p.description}")


def abrir(puerto: str, baud: int = 115200):
    """Abre el puerto y deja la placa corriendo, no retenida en reset.

    pyserial afirma DTR y RTS al abrir. En las placas con CH340 esas dos lineas
    van a EN y a GPIO0 a traves del circuito de auto-reset, asi que abrir el
    puerto "normalmente" puede dejar al ESP32 en reset permanente o entrando al
    bootloader: el firmware nunca arranca y el puerto queda mudo. Sintoma
    tipico, y desconcertante, es cero bytes recibidos con un firmware que
    funciona perfectamente.

    La secuencia de abajo suelta DTR, pulsa RTS para dar un reset limpio por EN,
    y lo suelta: la placa arranca la aplicacion normal.
    """
    try:
        import serial
    except ImportError:
        sys.exit("Falta pyserial.  pip install pyserial")
    try:
        s = serial.Serial(puerto, baud, timeout=1)
    except Exception as e:
        sys.exit(f"No se pudo abrir {puerto}: {e}")

    s.setDTR(False)
    s.setRTS(True)
    time.sleep(0.15)
    s.setRTS(False)
    time.sleep(0.35)  # margen para que arranque el bootloader y luego setup()
    s.reset_input_buffer()
    return s


def _leer_banner(ser, segundos: float = 3.0) -> list[str]:
    """Consume la cabecera de arranque, que incluye el escaneo del bus I2C."""
    fin = time.time() + segundos
    lineas = []
    while time.time() < fin:
        raw = ser.readline()
        if not raw:
            continue
        txt = raw.decode("utf-8", errors="replace").rstrip()
        if txt:
            lineas.append(txt)
            print(f"  {txt}")
    return lineas


def verificar(puerto: str) -> None:
    """Prueba de humo del hardware: que responde en el bus y llegan muestras."""
    ser = abrir(puerto)
    print("Arranque del dispositivo:")
    banner = _leer_banner(ser, 3.0)

    if not any("0x69" in ln for ln in banner):
        print("\n  AVISO: no se vio 0x69 en el escaneo. Sin IMU no hay captura.")
    if any("AD0 esta BAJO" in ln for ln in banner):
        print("\n  AVISO: el MPU6050 respondio en 0x68. La placa no es la esperada.")

    print("\nCapturando 3 s para medir la tasa real...")
    ser.write(b"s")
    t0 = time.time()
    n, primero, ultimo = 0, None, None
    while time.time() - t0 < 3.0:
        raw = ser.readline()
        if not raw:
            continue
        txt = raw.decode("utf-8", errors="replace").strip()
        if not txt or txt.startswith("#"):
            continue
        partes = txt.split(",")
        if len(partes) != len(COLUMNS):
            continue
        try:
            t_us = int(partes[1])
        except ValueError:
            continue
        if primero is None:
            primero = t_us
        ultimo = t_us
        n += 1
    ser.write(b"x")
    ser.close()

    if n < 2 or primero is None or ultimo is None:
        print(f"  FALLO: solo llegaron {n} muestras. Revisar el firmware.")
        return

    span = (ultimo - primero) / 1e6
    fs = (n - 1) / span if span > 0 else 0.0
    print(f"  {n} muestras en {span:.2f} s  ->  {fs:.2f} Hz  (objetivo {FS_HZ})")
    err = abs(fs - FS_HZ) / FS_HZ * 100
    if err < 1.0:
        print(f"  OK: desviacion {err:.2f} %. El muestreo es determinista.")
    elif err < 5.0:
        print(f"  ACEPTABLE: desviacion {err:.2f} %. Revisar carga del bus.")
    else:
        print(f"  PROBLEMA: desviacion {err:.2f} %. El jitter va a ensuciar el espectro.")


def grabar(args) -> None:
    if args.esfuerzo:
        guion = PROTOCOLO_ESFUERZO
    elif args.protocolo:
        guion = PROTOCOLO
    else:
        guion = [(args.tecnica, args.duracion, TECNICAS.get(args.tecnica, ""))]
    total = sum(d for _, d, _ in guion)

    print("=" * 66)
    print(f"  Sujeto {args.sujeto}   |   {len(guion)} bloques   |   "
          f"{total // 60}:{total % 60:02d} de grabacion")
    print("=" * 66)
    print(f"  Colocacion: {args.colocacion}")
    print(f"  Postura:    {args.postura}")
    print()
    print("  El sensor va sobre el ESTERNON SUPERIOR, plano contra el pecho.")
    print("  Si se coloca mas abajo, el cociente inclinacion/traslacion")
    print("  cambia de significado y los datos no sirven para entrenar.")
    print()
    input("  Enter cuando el sujeto este listo...")

    ser = abrir(args.puerto)
    _leer_banner(ser, 2.0)

    sesion = Sesion(
        sujeto=args.sujeto,
        fs_hz=FS_HZ,
        inicio_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        colocacion=args.colocacion,
        postura=args.postura,
        notas=args.notas,
        operador=args.operador,
    )

    DATA.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modo = "esfuerzo" if args.esfuerzo else ("protocolo" if args.protocolo else args.tecnica)
    base = f"{args.sujeto}_{modo}_{stamp}"
    csv_path = DATA / f"{base}.csv"
    json_path = DATA / f"{base}.json"

    filas: list[str] = []
    ser.reset_input_buffer()
    ser.write(b"s")
    t_inicio = time.time()
    n_total = 0

    try:
        for i, (tecnica, dur, consigna) in enumerate(guion, start=1):
            t_bloque = time.time()
            ini_rel = t_bloque - t_inicio
            print()
            print("-" * 66)
            print(f"  BLOQUE {i}/{len(guion)}   {tecnica.upper()}   {dur} s")
            print(f"  {consigna}")
            print("-" * 66)

            while True:
                transcurrido = time.time() - t_bloque
                if transcurrido >= dur:
                    break
                raw = ser.readline()
                if not raw:
                    continue
                txt = raw.decode("utf-8", errors="replace").strip()
                if not txt or txt.startswith("#"):
                    continue
                if txt.count(",") == len(COLUMNS) - 1:
                    filas.append(f"{txt},{tecnica}")
                    n_total += 1
                if n_total % 25 == 0:
                    resta = dur - transcurrido
                    print(f"\r  {resta:5.1f} s restantes   {n_total} muestras",
                          end="", flush=True)

            sesion.bloques.append(Bloque(tecnica, round(ini_rel, 2),
                                         round(time.time() - t_inicio, 2), consigna))
            print(f"\r  bloque completo                          ")

    except KeyboardInterrupt:
        print("\n\n  Interrumpido. Se guarda lo capturado hasta ahora.")
    finally:
        ser.write(b"x")
        time.sleep(0.3)
        ser.close()

    sesion.n_muestras = n_total
    sesion.duracion_s = round(time.time() - t_inicio, 2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# sujeto={sesion.sujeto} fs={FS_HZ} inicio={sesion.inicio_utc}\n")
        f.write(f"# colocacion={sesion.colocacion} postura={sesion.postura}\n")
        f.write("# valores CRUDOS int16 del registro 0x3B del MPU6050\n")
        f.write("# accel +/-2g = 16384 LSB/g   gyro +/-250dps = 131 LSB/dps\n")
        f.write(",".join(COLUMNS + ["tecnica"]) + "\n")
        f.write("\n".join(filas) + "\n")

    json_path.write_text(json.dumps(asdict(sesion), indent=2, ensure_ascii=False),
                         encoding="utf-8")

    esperado = sesion.duracion_s * FS_HZ
    perdida = 100.0 * (1 - n_total / esperado) if esperado > 0 else 0.0
    print()
    print("=" * 66)
    print(f"  {csv_path.name}")
    print(f"  {n_total} muestras   {sesion.duracion_s:.1f} s   "
          f"tasa efectiva {n_total / max(sesion.duracion_s, 1e-9):.1f} Hz")
    if perdida > 2.0:
        print(f"  AVISO: se perdio {perdida:.1f} % de las muestras esperadas.")
    print("=" * 66)
    print()
    print("  Siguiente:  python tools/analizar_captura.py "
          f"data/raw/{csv_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listar-puertos", action="store_true")
    ap.add_argument("--verificar", action="store_true",
                    help="prueba de humo del hardware y medicion de la tasa real")
    ap.add_argument("--puerto", help="COM5, /dev/ttyUSB0, ...")
    ap.add_argument("--sujeto", help="identificador anonimo, p.ej. s01")
    ap.add_argument("--tecnica", choices=sorted(TECNICAS), default="diaphragmatic")
    ap.add_argument("--duracion", type=int, default=120, help="segundos (sin --protocolo)")
    ap.add_argument("--protocolo", action="store_true",
                    help="guion completo de 9 bloques, 13 min")
    ap.add_argument("--esfuerzo", action="store_true",
                    help="guion de esfuerzo de 6 bloques, 6 min: separa tecnica "
                         "de intensidad")
    ap.add_argument("--colocacion", default="esternon superior")
    ap.add_argument("--postura", default="sentado erguido")
    ap.add_argument("--operador", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    if args.listar_puertos:
        listar_puertos()
        return
    if not args.puerto:
        ap.error("hace falta --puerto (usar --listar-puertos para verlos)")
    if args.verificar:
        verificar(args.puerto)
        return
    if not args.sujeto:
        ap.error("hace falta --sujeto")
    grabar(args)


if __name__ == "__main__":
    main()
