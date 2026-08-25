/* pneumocoach_config.h -- shared signal-chain contract
 *
 * GENERATED FILE -- DO NOT EDIT.
 * Produced by ml/scripts/emit_c_artifacts.py from ml/pneumocoach/config.py
 * Generated 2026-08-23T19:39:30Z
 *
 * Edit config.py and rerun the emitter instead. Hand edits here are
 * silently destroyed on the next build and, worse, make the firmware
 * disagree with the model it is running.
 */

#ifndef PNEUMOCOACH_CONFIG_H
#define PNEUMOCOACH_CONFIG_H

#include <stdint.h>

/* ---- I2C bus (MYOSA Mini Kit) ---- */
#define PC_I2C_SDA_PIN            21
#define PC_I2C_SCL_PIN            22
#define PC_I2C_FREQ_HZ            400000
/* MYOSA ties MPU6050 AD0 high. The usual 0x68 will NOT ACK. */
#define PC_ADDR_MPU6050           0x69
#define PC_ADDR_APDS9960          0x39
#define PC_ADDR_BMP180            0x77
#define PC_ADDR_SSD1306           0x3C

/* ---- Sensor scaling ---- */
#define PC_ACCEL_LSB_PER_G        16384.000000f
#define PC_GYRO_LSB_PER_DPS       131.072000f

/* ---- Framing ---- */
#define PC_FS_HZ                  50.0f
#define PC_DT_MS                  20
#define PC_WINDOW_N               600
#define PC_HOP_N                  150
#define PC_DECIM                  10
#define PC_DEC_N                  60
#define PC_NFFT                   128
#define PC_FS_DEC_HZ              5.000000f
#define PC_COMP_ALPHA             0.980000f
#define PC_WARMUP_N               1000

/* ---- Spectral bands (Hz) ---- */
#define PC_BAND_SLOW_LO           0.1000f
#define PC_BAND_SLOW_HI           0.2500f
#define PC_BAND_NORMAL_LO           0.2500f
#define PC_BAND_NORMAL_HI           0.5000f
#define PC_BAND_FAST_LO           0.5000f
#define PC_BAND_FAST_HI           1.0000f
#define PC_HF_ARTIFACT_HZ         2.0000f

/* ---- Model ---- */
#define PC_N_FEATURES             29
#define PC_N_CLASSES              3
#define PC_TENSOR_ARENA_BYTES     8192
#define PC_CONFIDENCE_FLOOR       0.6000f

/* ---- Per-session calibration ---- */
/* The device runs the same two reference manoeuvres as
 * ml/pneumocoach/calibracion.py and must use the same numbers.
 * Hand-copying either value into C is how the two drift apart. */
#define PC_REF_SEGUNDOS           30.0f
#define PC_REF_VENTANAS           6  /* minimo garantizado; puede salir una mas */
#define PC_CONTRASTE_MINIMO       0.001f

/* ---- Session pacing: the device advances on its own ---- */
/* Dwell times, not measurements. See config.py. */
#define PC_PREPARA_SEGUNDOS       10.0f
#define PC_RESULTADO_SEGUNDOS     8.0f
/* ---- TFLM operator set ----
 * The exported graph contains exactly these ops (confirmed with
 * tf.lite.experimental.Analyzer). Register only these:
 *
 *     MicroMutableOpResolver<2> resolver;
 *     resolver.AddFullyConnected();
 *     resolver.AddSoftmax();
 *
 * There is no RESHAPE and no QUANTIZE/DEQUANTIZE: input and output are
 * both int8, so the converter emitted no float conversion wrappers.
 * Pulling in AllOpsResolver instead costs flash for nothing.
 */
#define PC_TFLM_OP_COUNT          2

/* ---- Class table ---- */
typedef enum {
    PC_CLASS_DIAPHRAGMATIC    = 0,
    PC_CLASS_THORACIC         = 1,
    PC_CLASS_RAPID_SHALLOW    = 2,
} pc_class_t;

static const char *const PC_CLASS_KEY[3] = {
    "diaphragmatic",
    "thoracic",
    "rapid_shallow",
};
static const char *const PC_CLASS_OLED[3] = {
    "DIAFRAGMATICA",
    "TORACICA",
    "RAPIDA SUPERF.",
};
static const uint8_t PC_CLASS_COACHED_OK[3] = {
    1, 0, 0
};

/* ---- Feature order (load-bearing: the model was trained on it) ---- */
static const char *const PC_FEATURE_NAME[29] = {
    "tilt_rms",
    "tilt_p2p",
    "tilt_zcr",
    "tilt_dom_freq",
    "tilt_dom_ratio",
    "tilt_centroid",
    "tilt_bp_slow",
    "tilt_bp_normal",
    "tilt_bp_fast",
    "tilt_spec_entropy",
    "axial_rms",
    "axial_p2p",
    "axial_zcr",
    "axial_dom_freq",
    "axial_dom_ratio",
    "axial_centroid",
    "axial_bp_slow",
    "axial_bp_normal",
    "axial_bp_fast",
    "axial_spec_entropy",
    "log_tilt_axial_ratio",
    "tilt_axial_xcorr",
    "breath_rate_bpm",
    "breath_period_cv",
    "ie_ratio_mean",
    "ie_ratio_cv",
    "hf_energy_ratio",
    "jerk_max",
    "gyro_rms",
};

/* ---- Butterworth band-pass, 0.10-1.00 Hz @50 Hz, as SOS biquads ----
 * Layout per section: {b0, b1, b2, a1, a2} with a0 normalised to 1.
 * Direct Form II transposed, matching scipy.signal.sosfilt.
 */
#define PC_BP_SECTIONS            2
static const float PC_BP_SOS[10] = {
    0.00295827029f, 0.00591654058f, 0.00295827029f, -1.85429689f, 0.866742673f,
    1.0f, -2.0f, 1.0f, -1.98302722f, 0.983212525f,
};

#define PC_HF_SECTIONS            1
static const float PC_HF_SOS[5] = {
    0.837089191f, -1.67417838f, 0.837089191f, -1.64745998f, 0.700896781f,
};

/* ---- Feature standardisation: z = (x - mean) / scale ----
 * Applied in C before the tensor reaches the interpreter, so a bad
 * verdict can be traced without decompiling the model.
 */
static const float PC_FEATURE_MEAN[29] = {
    -0.00767692784f, 0.115878262f, 1.70357203f, -11.0039406f,
    -0.642079413f, -0.533698261f, -1.42336679f, 0.096773237f,
    20.1126823f, -0.216654122f, 0.563964844f, 0.381703228f,
    0.319760233f, -20.8665771f, -0.250556141f, 1.71575534f,
    0.334135711f, 0.82665801f, 1.97071338f, 0.582659066f,
    0.473757088f, 1.66969943f, 1.53530931f, -0.561196983f,
    1.41460741f, 3.9174993f, 2.53681707f, 0.868899524f,
    0.215932637f,
};

static const float PC_FEATURE_SCALE[29] = {
    1.50738716f, 1.21933794f, 4.85643864f, 34.6553535f,
    3.71493769f, 6.33255768f, 27.6999836f, 3.37439942f,
    65.2072906f, 2.50998402f, 1.62308812f, 1.34546638f,
    1.47472703f, 78.2543793f, 10.232214f, 5.70409632f,
    3.74220467f, 2.50544357f, 67.7107544f, 4.36569643f,
    0.836325765f, 6.42677593f, 6.95104361f, 3.87756801f,
    7.93911791f, 31.4744186f, 5.165874f, 1.49937701f,
    1.13282144f,
};

/* ---- Hann window for the decimated FFT ---- */
static const float PC_HANN_DEC[60] = {
    0.0f, 0.00283259989f, 0.0112983051f, 0.0253011958f,
    0.0446826136f, 0.0692229593f, 0.098644181f, 0.132612925f,
    0.17074431f, 0.212606295f, 0.257724565f, 0.305587912f,
    0.355654026f, 0.407355638f, 0.460106947f, 0.513310261f,
    0.566362764f, 0.61866335f, 0.669619433f, 0.71865366f,
    0.765210454f, 0.808762307f, 0.848815761f, 0.884916992f,
    0.91665696f, 0.943676038f, 0.965668089f, 0.982383934f,
    0.993634177f, 0.999291348f, 0.999291348f, 0.993634177f,
    0.982383934f, 0.965668089f, 0.943676038f, 0.91665696f,
    0.884916992f, 0.848815761f, 0.808762307f, 0.765210454f,
    0.71865366f, 0.669619433f, 0.61866335f, 0.566362764f,
    0.513310261f, 0.460106947f, 0.407355638f, 0.355654026f,
    0.305587912f, 0.257724565f, 0.212606295f, 0.17074431f,
    0.132612925f, 0.098644181f, 0.0692229593f, 0.0446826136f,
    0.0253011958f, 0.0112983051f, 0.00283259989f, 0.0f,
};

#endif /* PNEUMOCOACH_CONFIG_H */
