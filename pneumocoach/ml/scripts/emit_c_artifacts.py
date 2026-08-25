"""Render the Python contract into C sources the firmware compiles directly.

Emits into firmware/include/ and firmware/src/:

  pneumocoach_config.h    every shared constant, the biquad coefficients, the
                          standardiser vectors and the class table
  pneumocoach_model.h/.c  the INT8 TFLite flatbuffer as a const byte array
  pneumocoach_golden.h    reference feature vectors and expected verdicts

Nothing here is hand-maintained. If a constant changes in config.py, rerun this
script and the firmware picks it up; there is no second place to edit and
therefore no way for the two halves to disagree. The golden vectors are the
enforcement mechanism -- the on-device test replays them through the C DSP and
fails the build if the features drift from what Python computed.

Run:  python scripts/emit_c_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "tools"))
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _consola  # noqa: E402,F401  UTF-8 en Windows

from pneumocoach import config as C  # noqa: E402
from pneumocoach import dsp, synth  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "ml" / "artifacts"
INC = ROOT / "firmware" / "include"
SRC = ROOT / "firmware" / "src"

N_GOLDEN = 12
# Misma tolerancia que tools/paridad.py: float32 contra float64 nunca
# coincide bit a bit, y cualquier desviacion mayor es de algoritmo.
PC_GOLDEN_RTOL = 2.0e-2
PC_GOLDEN_ATOL = 1.0e-4


def banner(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return textwrap.dedent(
        f"""\
        /* {name}
         *
         * GENERATED FILE -- DO NOT EDIT.
         * Produced by ml/scripts/emit_c_artifacts.py from ml/pneumocoach/config.py
         * Generated {stamp}
         *
         * Edit config.py and rerun the emitter instead. Hand edits here are
         * silently destroyed on the next build and, worse, make the firmware
         * disagree with the model it is running.
         */
        """
    )


def _lit(v: float) -> str:
    """Literal float valido en C++.

    `%.9g` de 1.0 devuelve "1", y concatenar la sufijo da "1f", que el
    compilador interpreta como entero seguido de un operador de literal de
    usuario y rechaza con `unable to find numeric literal operator`. Hay que
    garantizar el punto decimal.
    """
    s = f"{float(v):.9g}"
    if not any(c in s for c in ".eEnN"):
        s += ".0"
    return s + "f"


def c_float_array(name: str, values, per_line: int = 4) -> str:
    body = []
    vals = [_lit(v) for v in values]
    for i in range(0, len(vals), per_line):
        body.append("    " + ", ".join(vals[i : i + per_line]) + ",")
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(body) + "\n};\n"


# --------------------------------------------------------------------------


def emit_config_header(std_mean: np.ndarray, std_scale: np.ndarray,
                       placeholder: bool = False) -> str:
    out = [banner("pneumocoach_config.h -- shared signal-chain contract")]
    out.append("#ifndef PNEUMOCOACH_CONFIG_H\n#define PNEUMOCOACH_CONFIG_H\n")
    out.append("#include <stdint.h>\n")

    out.append("/* ---- I2C bus (MYOSA Mini Kit) ---- */")
    out.append(f"#define PC_I2C_SDA_PIN            {C.I2C_SDA_PIN}")
    out.append(f"#define PC_I2C_SCL_PIN            {C.I2C_SCL_PIN}")
    out.append(f"#define PC_I2C_FREQ_HZ            {C.I2C_FREQ_HZ}")
    out.append("/* MYOSA ties MPU6050 AD0 high. The usual 0x68 will NOT ACK. */")
    out.append(f"#define PC_ADDR_MPU6050           0x{C.I2C_ADDR_MPU6050:02X}")
    out.append(f"#define PC_ADDR_APDS9960          0x{C.I2C_ADDR_APDS9960:02X}")
    out.append(f"#define PC_ADDR_BMP180            0x{C.I2C_ADDR_BMP180:02X}")
    out.append(f"#define PC_ADDR_SSD1306           0x{C.I2C_ADDR_SSD1306:02X}\n")

    out.append("/* ---- Sensor scaling ---- */")
    out.append(f"#define PC_ACCEL_LSB_PER_G        {C.ACCEL_LSB_PER_G:.6f}f")
    out.append(f"#define PC_GYRO_LSB_PER_DPS       {C.GYRO_LSB_PER_DPS:.6f}f\n")

    out.append("/* ---- Framing ---- */")
    out.append(f"#define PC_FS_HZ                  {C.FS_HZ:.1f}f")
    out.append(f"#define PC_DT_MS                  {C.DT_MS}")
    out.append(f"#define PC_WINDOW_N               {C.WINDOW_N}")
    out.append(f"#define PC_HOP_N                  {C.HOP_N}")
    out.append(f"#define PC_DECIM                  {C.DECIM}")
    out.append(f"#define PC_DEC_N                  {C.DEC_N}")
    out.append(f"#define PC_NFFT                   {C.NFFT}")
    out.append(f"#define PC_FS_DEC_HZ              {C.FS_DEC_HZ:.6f}f")
    out.append(f"#define PC_COMP_ALPHA             {C.COMP_ALPHA:.6f}f")
    out.append(f"#define PC_WARMUP_N               {dsp.WARMUP_N}\n")

    out.append("/* ---- Spectral bands (Hz) ---- */")
    for nm, (lo, hi) in (
        ("SLOW", C.BAND_SLOW),
        ("NORMAL", C.BAND_NORMAL),
        ("FAST", C.BAND_FAST),
    ):
        out.append(f"#define PC_BAND_{nm}_LO{'':<11}{lo:.4f}f")
        out.append(f"#define PC_BAND_{nm}_HI{'':<11}{hi:.4f}f")
    out.append(f"#define PC_HF_ARTIFACT_HZ         {C.HF_ARTIFACT_HZ:.4f}f\n")

    out.append("/* ---- Model ---- */")
    out.append(f"#define PC_N_FEATURES             {C.N_FEATURES}")
    out.append(f"#define PC_N_CLASSES              {C.N_CLASSES}")
    out.append(f"#define PC_TENSOR_ARENA_BYTES     {C.TENSOR_ARENA_BYTES}")
    out.append(f"#define PC_CONFIDENCE_FLOOR       {C.CONFIDENCE_FLOOR:.4f}f\n")

    out.append("/* ---- Per-session calibration ---- */")
    out.append("/* The device runs the same two reference manoeuvres as")
    out.append(" * ml/pneumocoach/calibracion.py and must use the same numbers.")
    out.append(" * Hand-copying either value into C is how the two drift apart. */")
    out.append(f"#define PC_REF_SEGUNDOS           {_lit(C.REF_SEGUNDOS)}")
    # Ventanas completas garantizadas dentro de la maniobra. El limite de fase
    # no cae alineado con la rejilla de ventanas, asi que salen esta cantidad
    # o una mas; se emite la garantizada para que la pantalla no anuncie un
    # objetivo que a veces no se alcanza.
    _refv = int((C.REF_SEGUNDOS * C.FS_HZ - C.WINDOW_N) // C.HOP_N)
    out.append(f"#define PC_REF_VENTANAS           {_refv}"
               "  /* minimo garantizado; puede salir una mas */")
    out.append(f"#define PC_CONTRASTE_MINIMO       {_lit(C.CONTRASTE_MINIMO)}\n")
    out.append("/* ---- Session pacing: the device advances on its own ---- */")
    out.append("/* Dwell times, not measurements. See config.py. */")
    out.append(f"#define PC_PREPARA_SEGUNDOS       {_lit(C.PREPARA_SEGUNDOS)}")
    out.append(f"#define PC_RESULTADO_SEGUNDOS     {_lit(C.RESULTADO_SEGUNDOS)}")

    out.append("/* ---- TFLM operator set ----")
    out.append(" * The exported graph contains exactly these ops (confirmed with")
    out.append(" * tf.lite.experimental.Analyzer). Register only these:")
    out.append(" *")
    out.append(f" *     MicroMutableOpResolver<{len(C.TFLM_OPS)}> resolver;")
    for op in C.TFLM_OPS:
        camel = "".join(p.capitalize() for p in op.split("_"))
        out.append(f" *     resolver.Add{camel}();")
    out.append(" *")
    out.append(" * There is no RESHAPE and no QUANTIZE/DEQUANTIZE: input and output are")
    out.append(" * both int8, so the converter emitted no float conversion wrappers.")
    out.append(" * Pulling in AllOpsResolver instead costs flash for nothing.")
    out.append(" */")
    out.append(f"#define PC_TFLM_OP_COUNT          {len(C.TFLM_OPS)}\n")

    out.append("/* ---- Class table ---- */")
    out.append("typedef enum {")
    for c in C.CLASSES:
        out.append(f"    PC_CLASS_{c.key.upper():<16} = {c.idx},")
    out.append("} pc_class_t;\n")
    out.append(f"static const char *const PC_CLASS_KEY[{C.N_CLASSES}] = {{")
    for c in C.CLASSES:
        out.append(f'    "{c.key}",')
    out.append("};")
    out.append(f"static const char *const PC_CLASS_OLED[{C.N_CLASSES}] = {{")
    for c in C.CLASSES:
        out.append(f'    "{c.oled}",')
    out.append("};")
    out.append(f"static const uint8_t PC_CLASS_COACHED_OK[{C.N_CLASSES}] = {{")
    out.append("    " + ", ".join("1" if c.coached_ok else "0" for c in C.CLASSES))
    out.append("};\n")

    out.append("/* ---- Feature order (load-bearing: the model was trained on it) ---- */")
    out.append(f"static const char *const PC_FEATURE_NAME[{C.N_FEATURES}] = {{")
    for n in C.FEATURE_NAMES:
        out.append(f'    "{n}",')
    out.append("};\n")

    out.append("/* ---- Butterworth band-pass, 0.10-1.00 Hz @50 Hz, as SOS biquads ----")
    out.append(" * Layout per section: {b0, b1, b2, a1, a2} with a0 normalised to 1.")
    out.append(" * Direct Form II transposed, matching scipy.signal.sosfilt.")
    out.append(" */")
    out.append(f"#define PC_BP_SECTIONS            {len(dsp.SOS_BANDPASS)}")
    flat_bp = [v for s in dsp.SOS_BANDPASS for v in (s[0], s[1], s[2], s[4], s[5])]
    out.append(c_float_array("PC_BP_SOS", flat_bp, per_line=5))

    out.append(f"#define PC_HF_SECTIONS            {len(dsp.SOS_HF)}")
    flat_hf = [v for s in dsp.SOS_HF for v in (s[0], s[1], s[2], s[4], s[5])]
    out.append(c_float_array("PC_HF_SOS", flat_hf, per_line=5))

    out.append("/* ---- Feature standardisation: z = (x - mean) / scale ----")
    out.append(" * Applied in C before the tensor reaches the interpreter, so a bad")
    out.append(" * verdict can be traced without decompiling the model.")
    out.append(" */")
    if placeholder:
        # Identity normalisation looks perfectly healthy at runtime: the
        # firmware boots, the interpreter runs, and it emits confident
        # nonsense. Nothing downstream can detect it. So the compiler has to.
        out.append("/* NO TRAINED STANDARDISER PRESENT.")
        out.append(" *")
        out.append(" * These are identity placeholders (mean 0, scale 1), not the")
        out.append(" * vectors of a fitted StandardScaler. They exist so that the rest")
        out.append(" * of the shared contract -- filter coefficients, Hann window,")
        out.append(" * feature order -- can be regenerated for the acquisition and DSP")
        out.append(" * firmware, which needs none of this.")
        out.append(" *")
        out.append(" * Any translation unit that performs inference must #error on")
        out.append(" * PC_STANDARDISER_PLACEHOLDER. Normalising with identity does not")
        out.append(" * fail: it produces confident, wrong verdicts, which is the exact")
        out.append(" * silent failure this project exists to avoid.")
        out.append(" */")
        out.append("#define PC_STANDARDISER_PLACEHOLDER 1\n")
    out.append(c_float_array("PC_FEATURE_MEAN", std_mean))
    out.append(c_float_array("PC_FEATURE_SCALE", std_scale))

    out.append("/* ---- Hann window for the decimated FFT ---- */")
    out.append(c_float_array("PC_HANN_DEC", np.hanning(C.DEC_N)))

    out.append("#endif /* PNEUMOCOACH_CONFIG_H */")
    return "\n".join(out) + "\n"


def emit_model(blob: bytes) -> tuple[str, str]:
    header = banner("pneumocoach_model.h -- INT8 TFLite flatbuffer")
    header += textwrap.dedent(
        f"""\
        #ifndef PNEUMOCOACH_MODEL_H
        #define PNEUMOCOACH_MODEL_H

        #include <stddef.h>
        #include <stdint.h>

        #define PC_MODEL_LEN {len(blob)}

        extern const unsigned char pc_model_tflite[PC_MODEL_LEN];

        #endif /* PNEUMOCOACH_MODEL_H */
        """
    )

    lines = []
    for i in range(0, len(blob), 12):
        chunk = blob[i : i + 12]
        lines.append("    " + " ".join(f"0x{b:02x}," for b in chunk))
    body = banner("pneumocoach_model.c -- INT8 TFLite flatbuffer")
    # El .c vive en firmware/src/ y su header en firmware/include/, asi que
    # por nombre corto no resuelve desde su propia carpeta.
    body += '#include "../include/pneumocoach_model.h"\n\n'
    # 16-byte alignment is required by the flatbuffer reader; without it TFLM
    # fails at schema verification with a message that looks nothing like an
    # alignment problem.
    body += "__attribute__((aligned(16)))\n"
    body += f"const unsigned char pc_model_tflite[PC_MODEL_LEN] = {{\n"
    body += "\n".join(lines)
    body += "\n};\n"
    return header, body


def emit_golden(std_mean, std_scale, blob: bytes) -> str:
    """Vectores de referencia de la cadena de INFERENCIA.

        29 caracteristicas -> z -> estandarizado -> INT8 -> probabilidades

    NO incluyen el DSP a proposito. De cuentas crudas a caracteristicas se
    encarga tools/paridad.py, que inyecta grabaciones enteras y filtra de forma
    continua, que es como funciona el dispositivo.

    La primera version metia cuentas crudas aqui y comparaba las 29
    caracteristicas. No podia funcionar: eran ventanas SUELTAS, y Python las
    habia calculado sobre el filtrado continuo de toda la grabacion, con estado
    acumulado de minutos anteriores. El dispositivo arrancaba sus filtros en
    frio en cada ventana. Salian 190 de 348 caracteristicas fuera de tolerancia
    -y aun asi los 12 veredictos coincidian, que es exactamente el tipo de
    coincidencia que no hay que confundir con una verificacion-.

    Lo que estos vectores si verifican, y nadie mas verifica, es el tramo que
    va de la caracteristica al veredicto: la proyeccion sobre el eje del
    paciente, el estandarizado, la cuantizacion y el interprete, sobre el
    dispositivo y sin que nadie tenga que llevar el sensor puesto.
    """
    from pneumocoach.calibracion import ReferenciaSesion
    from pneumocoach.train import tflite_probabilities
    sys.path.insert(0, str(ROOT / "tools"))
    from analizar_captura import bloques, cargar, cargar_calibracion

    grabacion = ROOT / "data" / "raw" / "s01_protocolo_20260822_120903.csv"
    if not grabacion.exists():
        raise SystemExit(f"falta la grabacion de referencia: {grabacion}")

    import json as _json
    meta = _json.loads(grabacion.with_suffix(".json").read_text(encoding="utf-8"))
    R, sesgo = cargar_calibracion("s01", meta.get("inicio_utc"))
    counts, et = cargar(grabacion)
    ch = dsp.channels_from_counts(counts, mount=R, gyro_bias_dps=sesgo)

    ES_DIA = ("diaphragmatic", "dia_suave", "dia_fuerte")
    ES_TOR = ("thoracic", "tor_suave", "tor_fuerte")
    ref_n = int(C.REF_SEGUNDOS * C.FS_HZ)

    X_ref = {"dia": [], "tor": []}
    candidatos = {c.key: [] for c in C.CLASSES}
    for tec, a, b in bloques(et):
        for i in range(a, b - C.WINDOW_N + 1, C.HOP_N):
            f = dsp.extract_features({k: v[i:i + C.WINDOW_N] for k, v in ch.items()})
            en_ref = (i + C.WINDOW_N) - a <= ref_n
            if tec in ES_DIA and en_ref:
                X_ref["dia"].append(f)
            elif tec in ES_TOR and en_ref:
                X_ref["tor"].append(f)
            elif tec in [c.key for c in C.CLASSES] and not en_ref:
                candidatos[tec].append((i, counts[i:i + C.WINDOW_N], f))

    ref = ReferenciaSesion.desde_ventanas(np.stack(X_ref["dia"]),
                                          np.stack(X_ref["tor"]))

    # Reparto equilibrado entre clases, tomando ventanas separadas para que no
    # compartan datos entre si.
    por_clase = max(1, N_GOLDEN // C.N_CLASSES)
    feats = []
    for c in C.CLASSES:
        disponibles = candidatos[c.key]
        if not disponibles:
            continue
        paso = max(1, len(disponibles) // por_clase)
        for k in range(0, len(disponibles), paso):
            if len(feats) >= (c.idx + 1) * por_clase:
                break
            _, _, f = disponibles[k]
            feats.append(f)

    feat_arr = np.stack(feats)
    z = ref.normaliza(feat_arr).astype(np.float32)
    x = ((z - std_mean) / std_scale).astype(np.float32)
    probs = tflite_probabilities(blob, x)
    verdicts = probs.argmax(axis=1)

    n = len(feats)
    out = [banner("pneumocoach_golden.h -- Python/C parity vectors")]
    out.append("#ifndef PNEUMOCOACH_GOLDEN_H\n#define PNEUMOCOACH_GOLDEN_H\n")
    out.append("#include <stdint.h>\n")
    out.append(f"#define PC_GOLDEN_N {n}\n")
    out.append("/* Referencia de sesion con la que se calculo z. El dispositivo la")
    out.append(" * carga para reproducir exactamente la misma proyeccion. */")
    out.append(c_float_array("PC_GOLDEN_REF_DIA", ref.dia))
    out.append(c_float_array("PC_GOLDEN_REF_TOR", ref.tor))

    def matriz(nombre, M, tipo="float"):
        f2 = []
        for fila in M:
            vals = ", ".join(
                _lit(v) if tipo == "float" else str(int(v))
                for v in fila)
            f2.append("  {" + vals + "},")
        return (f"static const {tipo} {nombre}[PC_GOLDEN_N][{M.shape[1]}] = {{\n"
                + "\n".join(f2) + "\n};\n")

    out.append(matriz("PC_GOLDEN_FEATURES", feat_arr))
    out.append(matriz("PC_GOLDEN_Z", z))
    out.append(matriz("PC_GOLDEN_PROB", probs))
    out.append("static const uint8_t PC_GOLDEN_VERDICT[PC_GOLDEN_N] = {")
    out.append("    " + ", ".join(str(int(v)) for v in verdicts) + ",")
    out.append("};\n")
    out.append(f"#define PC_GOLDEN_RTOL {_lit(PC_GOLDEN_RTOL)}")
    out.append(f"#define PC_GOLDEN_ATOL {_lit(PC_GOLDEN_ATOL)}\n")
    out.append("#endif /* PNEUMOCOACH_GOLDEN_H */")
    return "\n".join(out) + "\n"


def main() -> None:
    model_path = ARTIFACTS / "model_int8.tflite"
    std_path = ARTIFACTS / "standardiser.npz"

    # El contrato compartido -constantes, coeficientes de filtro, ventana de
    # Hann, orden de caracteristicas- no depende del modelo y tiene que poder
    # regenerarse sin el. Desde que el modelo entrenado con fisica refutada se
    # movio a RETIRADO/, este script debe seguir sirviendo para el firmware.
    solo_config = not model_path.exists()
    if solo_config:
        print("AVISO: no hay model_int8.tflite. Se emite solo el contrato "
              "compartido; sin flatbuffer ni vectores dorados.")
        mean = np.zeros(C.N_FEATURES, dtype=np.float32)
        scale = np.ones(C.N_FEATURES, dtype=np.float32)
        blob = b""
    else:
        blob = model_path.read_bytes()
        npz = np.load(std_path)
        mean, scale = npz["mean"], npz["scale"]

    INC.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)

    (INC / "pneumocoach_config.h").write_text(
        emit_config_header(mean, scale, placeholder=solo_config))
    print(f"emitted to {INC}")
    print(f"  pneumocoach_config.h    {C.N_FEATURES} features, {C.N_CLASSES} classes")
    if solo_config:
        print("  PC_STANDARDISER_PLACEHOLDER definido: cualquier unidad de")
        print("  compilacion que haga inferencia debe fallar el build.")
        return
    mh, mc = emit_model(blob)
    (INC / "pneumocoach_model.h").write_text(mh)
    (SRC / "pneumocoach_model.c").write_text(mc)
    (INC / "pneumocoach_golden.h").write_text(emit_golden(mean, scale, blob))
    print(f"  pneumocoach_model.c     {len(blob)} bytes of flatbuffer")
    print(f"  pneumocoach_golden.h    {N_GOLDEN} parity windows")


if __name__ == "__main__":
    main()
