"""Single source of truth for the PneumoCoach signal chain.

Every constant that BOTH the Python training pipeline and the ESP32 firmware
must agree on lives here. `scripts/emit_config_header.py` renders this module
into `firmware/include/pneumocoach_config.h`, so the two can never drift.

Rule: if a number appears in both Python and C, it belongs in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Hardware (MYOSA Mini Kit -- these five modules only)
# --------------------------------------------------------------------------

I2C_ADDR_MPU6050 = 0x69  # MYOSA ties AD0 high; the usual 0x68 is WRONG here
I2C_ADDR_APDS9960 = 0x39
I2C_ADDR_BMP180 = 0x77
I2C_ADDR_SSD1306 = 0x3C

I2C_SDA_PIN = 21
I2C_SCL_PIN = 22
I2C_FREQ_HZ = 400_000  # Fast mode

# MPU6050 full-scale ranges we configure at boot.
ACCEL_FS_G = 2.0  # +/- 2 g   -> maximum sensitivity, breathing is sub-100 mg
GYRO_FS_DPS = 250.0  # +/- 250 dps
ACCEL_LSB_PER_G = 32768.0 / ACCEL_FS_G  # 16384 LSB/g
GYRO_LSB_PER_DPS = 32768.0 / GYRO_FS_DPS  # 131.07 LSB/(deg/s)

# Ruido del sensor.
#
# OJO: el modulo del kit dice "MPU-6050" impreso pero su WHO_AM_I devuelve 0x70,
# o sea silicio **MPU6500**. Verificado sobre la placa el 2026-08-17. Es habitual
# en los clones GY-521. Los registros de datos son compatibles; el ruido y la
# configuracion del anti-aliasing no lo son.
#
# Hoja de datos MPU6500: accel 300 ug/rtHz, gyro 0.01 dps/rtHz.
# Medido en banco (20 s, placa sobre escritorio, sin aislar de vibracion):
# desviacion muestra a muestra en az = 4.26 mg, lo que implica sigma ~3.0 mg en
# banda y una densidad efectiva de ~650 ug/rtHz: el doble de la hoja de datos.
#
# Se usa el valor MEDIDO, no el de catalogo, porque el modelo se entrena para
# funcionar sobre este hardware y no sobre el ideal. Queda pendiente repetir la
# medicion con la placa sobre espuma para separar ruido del sensor de vibracion
# del banco; si baja, se ajusta.
ACCEL_NOISE_DENSITY_UG_RTHZ = 650.0  # ug/sqrt(Hz)  -- medido, no de catalogo
GYRO_NOISE_DENSITY_DPS_RTHZ = 0.010  # dps/sqrt(Hz) -- MPU6500

# Sesgo medido en el ejemplar del kit. El giroscopo nunca marca cero exacto y
# el sesgo integra a deriva en el filtro complementario.
GYRO_BIAS_DPS_TIPICO = (0.28, -0.79, 0.24)

# La magnitud de la gravedad medida dio 1.063 g en vez de 1.000. Un 6% de error
# de escala/sesgo, dentro de lo esperable en un modulo sin calibrar pero
# suficiente para justificar una rutina de calibracion antes de grabar el
# dataset definitivo.
ACCEL_ESCALA_MEDIDA_G = 1.063

# --------------------------------------------------------------------------
# Sampling and framing
# --------------------------------------------------------------------------

FS_HZ = 50.0  # acquisition rate (ADR-0003)
DT_MS = 20  # FreeRTOS acquisition period

WINDOW_S = 12.0  # long enough for >=1 full breath at 6 BPM
HOP_S = 3.0  # 75% overlap -> a verdict every 3 s
WINDOW_N = int(WINDOW_S * FS_HZ)  # 600 samples
HOP_N = int(HOP_S * FS_HZ)  # 150 samples

# Respiratory band-pass. 0.1 Hz kills gravity/postural DC, 1.0 Hz kills
# ballistocardiographic and motion content.
BP_LOW_HZ = 0.10
BP_HIGH_HZ = 1.00
BP_ORDER = 2  # 2nd-order Butterworth, applied as cascaded biquads in C

# Complementary filter weight for tilt fusion (accel vs gyro).
COMP_ALPHA = 0.98

# Spectral analysis runs on the band-passed signal decimated by 10.
DECIM = 10
FS_DEC_HZ = FS_HZ / DECIM  # 5 Hz
DEC_N = WINDOW_N // DECIM  # 60 samples
NFFT = 128  # zero-padded; 0.039 Hz bins, radix-2 friendly in C

# Spectral sub-bands, in Hz, and their breath-rate meaning.
BAND_SLOW = (0.10, 0.25)  # 6-15 BPM   coached diaphragmatic range
BAND_NORMAL = (0.25, 0.50)  # 15-30 BPM  resting/thoracic
BAND_FAST = (0.50, 1.00)  # 30-60 BPM  tachypnoeic
HF_ARTIFACT_HZ = 2.0  # energy above this = motion/cough, not breathing

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BreathClass:
    idx: int
    key: str
    label_en: str
    label_es: str
    oled: str  # <=16 chars, fits one SSD1306 text row at size 1
    coached_ok: bool  # is this the technique we want the patient to hold?


CLASSES: tuple[BreathClass, ...] = (
    BreathClass(0, "diaphragmatic", "Diaphragmatic", "Diafragmatica", "DIAFRAGMATICA", True),
    BreathClass(1, "thoracic", "Thoracic (shallow)", "Toracica (superficial)", "TORACICA", False),
    BreathClass(2, "rapid_shallow", "Rapid shallow", "Rapida superficial", "RAPIDA SUPERF.", False),
)

# `artifact` is NOT a class. Measured, not assumed: training it alongside the
# techniques costs 5 points on the question that matters (0.694 vs 0.750 binary
# across protocols), because a cough is so far from either technique that the
# boundary between them gets spent separating it instead. It also only exists in
# the long protocol, so it could never be validated across protocols anyway.
#
# `rapid_shallow` stays: adding it does not hurt (0.750 vs 0.744 without it) and
# tachypnoea is a clinically real state worth naming.

# There is deliberately NO "resting" class, and the device does NOT try to tell
# whether the patient is exercising at all.
#
# We measured it. A classifier asked to separate resting breathing from a
# deliberate manoeuvre scores 0.460 across protocols -- below chance. On the
# patient's own axis, resting sits at z = +0.86 on tilt_rms, essentially on top
# of thoracic (+0.84): breathing at rest genuinely looks like breathing with the
# chest, because for most people it partly is.
#
# So the verdict is only meaningful DURING A GUIDED EXERCISE, when the patient
# has been asked to perform a specific manoeuvre. Left running unattended the
# device would label ordinary breathing as a technique, and no confidence floor
# fixes that: 98 % of resting windows came out above it.

N_CLASSES = len(CLASSES)
CLASS_KEYS = tuple(c.key for c in CLASSES)
CLASS_INDEX = {c.key: c.idx for c in CLASSES}

# Below this softmax score the firmware shows "---" instead of a verdict.
# Refusing to guess is better than coaching the patient wrong.
CONFIDENCE_FLOOR = 0.60

# --------------------------------------------------------------------------
# Per-session calibration
# --------------------------------------------------------------------------
# Lives here, not in calibracion.py, because the firmware runs the same
# calibration on-device and both sides must agree on the numbers. A hand-copied
# constant in C is how the two implementations drift apart.

# Length of each reference manoeuvre. Three slow breaths fit in 30 s even at
# 6 per minute, the slow end of the coached range.
REF_SEGUNDOS = 30.0

# If the patient performs both manoeuvres almost identically the axis degenerates
# and the division blows up. Below this relative contrast the feature is marked
# non-informative for that session and neutralised.
CONTRASTE_MINIMO = 1e-3

# --------------------------------------------------------------------------
# Session pacing
# --------------------------------------------------------------------------
# The session advances on its own: no button, no gesture, no PC. The patient is
# mid-exercise with both hands on their body, so anything they have to operate
# is a movement that lands in the channel we measure.
#
# Unlike everything else in this file, these two are NOT measured quantities.
# They are dwell times chosen so the on-screen instruction can be read without
# hurrying, and they are declared as such rather than dressed up with a
# derivation they do not have. If a rehearsal shows they are too short, the
# thing to change is the number, not the reasoning.
PREPARA_SEGUNDOS = 10.0    # LISTO -> first reference manoeuvre
RESULTADO_SEGUNDOS = 8.0   # CALIBRADO -> coaching

# CONTRASTE_MINIMO above is a numerical guard, not a quality check: it only
# stops a division by an axis that is essentially zero. Noise alone clears it,
# so a session where the patient performed both manoeuvres identically still
# reports all 29 features as having an axis.
#
# There is deliberately NO quality threshold here. We looked for one --
# tools/medir_calidad_calibracion.py tests both the relative contrast and
# Cohen's d against a null built from two halves of the same manoeuvre -- and
# neither separates: the ranges overlap, so no threshold accepts every good
# calibration while rejecting any bad one. A threshold added anyway would be
# false reassurance. The device shows the number and gates nothing.

# --------------------------------------------------------------------------
# Feature vector -- ORDER IS LOAD-BEARING
# --------------------------------------------------------------------------
# The C extractor must emit these in exactly this order. `tests/test_parity.py`
# and the firmware golden-vector test both enforce it.

_PER_CHANNEL_FEATURES = (
    ("rms", "RMS amplitude of the band-passed channel"),
    ("p2p", "Peak-to-peak excursion"),
    ("zcr", "Zero-crossing rate (Hz)"),
    ("dom_freq", "Dominant spectral frequency (Hz)"),
    ("dom_ratio", "Dominant-bin power / total power (spectral purity)"),
    ("centroid", "Spectral centroid (Hz)"),
    ("bp_slow", "Normalised band power 0.10-0.25 Hz"),
    ("bp_normal", "Normalised band power 0.25-0.50 Hz"),
    ("bp_fast", "Normalised band power 0.50-1.00 Hz"),
    ("spec_entropy", "Normalised spectral entropy"),
)

CHANNELS = ("tilt", "axial")


def _build_feature_spec() -> tuple[tuple[str, str], ...]:
    spec: list[tuple[str, str]] = []
    for ch in CHANNELS:
        for name, doc in _PER_CHANNEL_FEATURES:
            spec.append((f"{ch}_{name}", f"{ch}: {doc}"))
    spec += [
        # Cross-channel: the single most discriminative pair. Thoracic
        # breathing rotates the upper chest (tilt-dominant); diaphragmatic
        # breathing translates it antero-posteriorly (axial-dominant).
        ("log_tilt_axial_ratio", "log10(tilt_rms / axial_rms)"),
        ("tilt_axial_xcorr", "Zero-lag Pearson correlation tilt vs axial"),
        # Breath-cycle timing, from hysteresis segmentation of the tilt
        # channel -- the higher-SNR of the two in every class (see synth.py).
        ("breath_rate_bpm", "Mean breath rate (breaths/min)"),
        ("breath_period_cv", "Coefficient of variation of breath period"),
        ("ie_ratio_mean", "Mean inspiration:expiration duration ratio"),
        ("ie_ratio_cv", "Coefficient of variation of the I:E ratio"),
        # Artifact detectors, computed on the RAW (unfiltered) stream.
        ("hf_energy_ratio", "Fraction of accel energy above 2 Hz"),
        ("jerk_max", "Max |d(accel)/dt| (g/s)"),
        ("gyro_rms", "RMS gyro magnitude (dps)"),
    ]
    return tuple(spec)


FEATURE_SPEC = _build_feature_spec()
FEATURE_NAMES = tuple(name for name, _ in FEATURE_SPEC)
N_FEATURES = len(FEATURE_NAMES)  # 29

# --------------------------------------------------------------------------
# Model / memory budget (ADR-0004). CI fails the build if these are exceeded.
# --------------------------------------------------------------------------

MLP_HIDDEN = (32, 16)

# Provisional. Verified by ModelAnalyzer, the graph holds only ~48 bytes of
# activations (24 + 16 + 4 + 4 int8), so the arena is dominated by TFLM
# bookkeeping rather than tensor data. Start here, then shrink to whatever
# `interpreter.arena_used_bytes()` reports after AllocateTensors() plus ~25%.
TENSOR_ARENA_BYTES = 8 * 1024

MAX_MODEL_BYTES = 8 * 1024  # INT8 TFLite flatbuffer ceiling
MAX_INFERENCE_MS = 10.0  # must fit inside one 20 ms acquisition tick
SRAM_HEADROOM_MIN_PCT = 15.0  # PM blocker threshold

# Exactly the operators present in the exported graph, confirmed with
# tf.lite.experimental.Analyzer:
#     FULLY_CONNECTED x3, SOFTMAX x1
# No RESHAPE and no QUANTIZE/DEQUANTIZE, because inference_input_type and
# inference_output_type are both int8 -- the converter would otherwise wrap the
# graph in float conversion ops. Register ONLY these two in the firmware:
#     MicroMutableOpResolver<2> resolver;
#     resolver.AddFullyConnected();
#     resolver.AddSoftmax();
# Registering the full op set instead would drag the whole kernel library into
# flash for no benefit. Re-run the analyzer after any architecture change.
TFLM_OPS = ("FULLY_CONNECTED", "SOFTMAX")


@dataclass(frozen=True)
class Budget:
    """Resource envelope tracked in docs/resource-budget.md."""

    tensor_arena_bytes: int = TENSOR_ARENA_BYTES
    max_model_bytes: int = MAX_MODEL_BYTES
    max_inference_ms: float = MAX_INFERENCE_MS
    sram_headroom_min_pct: float = SRAM_HEADROOM_MIN_PCT
    notes: tuple[str, ...] = field(default_factory=tuple)


BUDGET = Budget()
