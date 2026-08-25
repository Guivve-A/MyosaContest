"""Respiratory DSP chain -- the Python reference for the C implementation.

This module is deliberately written the way the firmware will run, not the way
NumPy makes convenient:

  * Filtering is **causal** (`sosfilt`, single pass). Using `sosfiltfilt` here
    would train the model on zero-phase data the ESP32 can never produce -- a
    textbook train/serve skew. The price is a settling transient, which we pay
    once per recording by discarding a warm-up prefix.

  * Filters run **continuously over the whole stream**, and windows are cut
    from the already-filtered signal. That is what a streaming device does.
    Filtering each window independently would put a settling transient inside
    every single feature vector.

  * Decimation before the FFT is plain subsampling, not `scipy.signal.decimate`.
    The band-pass already limits content to 1 Hz, comfortably under the 2.5 Hz
    Nyquist of the 5 Hz decimated rate, so an extra anti-alias stage would be
    dead code in C -- and a silent mismatch if Python had one and C did not.

`tests/test_parity.py` pins the numeric behaviour so the C port can be checked
against it byte for byte.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import config as C

EPS = 1e-12


# --------------------------------------------------------------------------
# Filter design -- computed once, exported to C as biquad coefficients
# --------------------------------------------------------------------------


def _design_bandpass() -> np.ndarray:
    return signal.butter(
        C.BP_ORDER,
        [C.BP_LOW_HZ, C.BP_HIGH_HZ],
        btype="bandpass",
        fs=C.FS_HZ,
        output="sos",
    )


def _design_hf_highpass() -> np.ndarray:
    """2 Hz high-pass used only to measure motion/speech contamination."""
    return signal.butter(2, C.HF_ARTIFACT_HZ, btype="highpass", fs=C.FS_HZ, output="sos")


SOS_BANDPASS = _design_bandpass()
SOS_HF = _design_hf_highpass()

# How much of each recording to throw away while the 0.1 Hz high-pass settles.
#
# Measured, not guessed -- `tools/medir_asentamiento.py` reports 12.4 s worst
# case across the recordings on hand, so this carries a 1.6x margin. It used to
# be 60 s, which was a guess made before the filters were seeded and which threw
# away three quarters of a minute of every session for nothing.
#
# It cannot go much lower: a 0.1 Hz high-pass has roughly ten seconds of memory,
# so ten seconds of real signal is what it takes to know where the baseline is.
# That part is the filter's physics, not an initial condition, and no amount of
# seeding removes it.
WARMUP_S = 20.0
WARMUP_N = int(WARMUP_S * C.FS_HZ)


# --------------------------------------------------------------------------
# Stage 1: raw counts -> the two mechanical channels
# --------------------------------------------------------------------------


def complementary_pitch(
    accel_g: np.ndarray, gyro_dps: np.ndarray, alpha: float = C.COMP_ALPHA, fs: float = C.FS_HZ
) -> np.ndarray:
    """Fuse accel and gyro into a drift-free chest-wall pitch angle (degrees).

    The accelerometer gives an absolute but noisy and motion-corrupted angle;
    the gyro gives a clean rate that integrates into drift. The complementary
    filter takes the low frequencies from one and the high from the other.

    The firmware runs the scalar recurrence

        p[k] = alpha * (p[k-1] + gy[k]*dt) + (1-alpha) * pitch_acc[k]

    which is a one-pole IIR in disguise: with u[k] = alpha*gy[k]*dt +
    (1-alpha)*pitch_acc[k] it collapses to p[k] = alpha*p[k-1] + u[k]. We
    evaluate it with `lfilter` for speed; `reference_complementary_pitch`
    below is the literal loop, and `tests/test_parity.py` asserts the two
    agree to float64 rounding.
    """
    pitch_acc, u = _comp_terms(accel_g, gyro_dps, alpha, fs)
    # Seed the state with the accelerometer angle, not zero, so the estimate
    # starts on the body instead of walking up from horizontal.
    zi = np.array([alpha * pitch_acc[0]])
    out, _ = signal.lfilter([1.0], [1.0, -alpha], u, zi=zi)
    return out


def _comp_terms(
    accel_g: np.ndarray, gyro_dps: np.ndarray, alpha: float, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    ax, ay, az = accel_g[:, 0], accel_g[:, 1], accel_g[:, 2]
    gy = gyro_dps[:, 1]
    pitch_acc = np.degrees(np.arctan2(az, np.sqrt(ax * ax + ay * ay) + EPS))
    u = alpha * gy * (1.0 / fs) + (1.0 - alpha) * pitch_acc
    return pitch_acc, u


def reference_complementary_pitch(
    accel_g: np.ndarray, gyro_dps: np.ndarray, alpha: float = C.COMP_ALPHA, fs: float = C.FS_HZ
) -> np.ndarray:
    """Literal sample-by-sample form -- the spec the C code implements."""
    pitch_acc, _ = _comp_terms(accel_g, gyro_dps, alpha, fs)
    gy = gyro_dps[:, 1]
    dt = 1.0 / fs
    out = np.empty(len(accel_g), dtype=np.float64)
    p = pitch_acc[0]
    for k in range(len(out)):
        p = alpha * (p + gy[k] * dt) + (1.0 - alpha) * pitch_acc[k]
        out[k] = p
    return out


def channels_from_counts(
    counts: np.ndarray,
    fs: float = C.FS_HZ,
    mount: np.ndarray | None = None,
    gyro_bias_dps: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Full front end: int16 registers -> filtered tilt and axial channels.

    `mount` is the 3x3 rotation from `tools/orientacion.py`, with rows
    [superior; lateral; anterior], so that `v_body = R @ v_sensor`. Pass it and
    everything downstream reasons in the anatomical frame no matter how the
    module was actually taped on. Omit it and the sensor frame is assumed to be
    the anatomical one already, which is what the synthetic generator produces.

    Measuring the mount rather than mandating it is what lets the enclosure be
    designed for comfort: a board rotated 45 degrees on the sternum -- which is
    what the first real subject produced -- is fully recoverable, but only if
    the rotation is known.

    `gyro_bias_dps` is subtracted before integration. An uncorrected bias of
    ~1 dps walks the complementary filter's angle estimate by a degree every
    second, which swamps a breathing signal of one to four degrees.
    """
    phys = counts.astype(np.float64)
    phys[:, :3] /= C.ACCEL_LSB_PER_G
    phys[:, 3:] /= C.GYRO_LSB_PER_DPS
    accel_g, gyro_dps = phys[:, :3], phys[:, 3:]

    if gyro_bias_dps is not None:
        gyro_dps = gyro_dps - np.asarray(gyro_bias_dps, dtype=np.float64)

    if mount is not None:
        R = np.asarray(mount, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"mount debe ser 3x3, no {R.shape}")
        accel_g = accel_g @ R.T
        gyro_dps = gyro_dps @ R.T

    pitch_deg = complementary_pitch(accel_g, gyro_dps, fs=fs)

    # Subtract the gravity projection so what remains on Z is true antero-
    # posterior translation rather than the tilt we already measured.
    az_trans = accel_g[:, 2] - np.sin(np.radians(pitch_deg))

    accel_mag = np.linalg.norm(accel_g, axis=1)

    # Filter the deviation from the first sample, not the absolute value.
    #
    # The band-pass input is the mount angle -- around 45 degrees on a real
    # torso -- and the interesting signal riding on it is a few tenths of a
    # degree. A high-pass has to cancel that pedestal to expose the breathing,
    # and on the ESP32, where this runs in float32, cancelling 45 against 45 to
    # recover 0.36 burns most of the available precision. Measured on-device,
    # the tilt channel drifted from the float64 reference by 1.3e-3; subtracting
    # the pedestal first brings it to 1.4e-4.
    #
    # It costs nothing in exactness. A high-pass has zero DC gain, so removing a
    # constant from its input cannot change its steady-state output -- the two
    # forms agree to 1.5e-12 in float64. What changes is only how much precision
    # float32 has left to work with, which is why the firmware does the same.
    tilt = signal.sosfilt(SOS_BANDPASS, pitch_deg - pitch_deg[0])
    axial = signal.sosfilt(SOS_BANDPASS, az_trans - az_trans[0])

    # The magnitude also goes in relative to its first sample. It used to have
    # the mean of the whole recording subtracted, which is non-causal -- the
    # mean of data that has not arrived yet -- and unreproducible on-device.
    hf = signal.sosfilt(SOS_HF, accel_mag - accel_mag[0])

    return {
        "tilt": tilt,
        "axial": axial,
        "accel_mag": accel_mag,
        "hf": hf,
        "gyro_mag": np.linalg.norm(gyro_dps, axis=1),
        "pitch": pitch_deg,
    }


# --------------------------------------------------------------------------
# Stage 2: spectral helpers (128-point FFT on the 5 Hz decimated signal)
# --------------------------------------------------------------------------

_HANN_DEC = np.hanning(C.DEC_N)
_FREQS = np.fft.rfftfreq(C.NFFT, d=1.0 / C.FS_DEC_HZ)


def spectrum(x: np.ndarray) -> np.ndarray:
    """Power spectrum of one window: decimate x10, Hann, zero-pad, rFFT."""
    dec = x[:: C.DECIM][: C.DEC_N]
    if len(dec) < C.DEC_N:
        dec = np.pad(dec, (0, C.DEC_N - len(dec)))
    dec = (dec - dec.mean()) * _HANN_DEC
    spec = np.fft.rfft(dec, n=C.NFFT)
    return (spec.real**2 + spec.imag**2).astype(np.float64)


def _band_power(power: np.ndarray, lo: float, hi: float) -> float:
    m = (_FREQS >= lo) & (_FREQS < hi)
    return float(power[m].sum())


def spectral_features(x: np.ndarray) -> dict[str, float]:
    power = spectrum(x)
    # Ignore the DC/near-DC bins: the band-pass already removed real content
    # there, so anything left is filter ringing.
    valid = _FREQS >= 0.05
    p = power[valid]
    f = _FREQS[valid]
    total = float(p.sum()) + EPS

    k = int(np.argmax(p))
    pn = p / total
    entropy = float(-(pn * np.log(pn + EPS)).sum() / np.log(len(pn)))

    return {
        "dom_freq": float(f[k]),
        "dom_ratio": float(p[k] / total),
        "centroid": float((f * p).sum() / total),
        "bp_slow": _band_power(power, *C.BAND_SLOW) / total,
        "bp_normal": _band_power(power, *C.BAND_NORMAL) / total,
        "bp_fast": _band_power(power, *C.BAND_FAST) / total,
        "spec_entropy": entropy,
    }


# --------------------------------------------------------------------------
# Stage 3: breath segmentation (Schmitt trigger -- trivially portable to C)
# --------------------------------------------------------------------------


def _instante_cruce(x: np.ndarray, k: int, thr: float) -> float:
    """Instante, en muestras, en que x cruza thr entre k-1 y k.

    Sin esto el instante del cruce se cuantiza a la muestra entera, y una
    muestra que cae a filo del umbral se decide por diferencias del orden de
    1e-4. Medido en el dispositivo: en una ventana la senal pasaba a 9.8e-5 del
    umbral mientras la diferencia float32/float64 entre las dos
    implementaciones era 1.06e-4 -mayor que la distancia al umbral-, asi que el
    cruce caia en muestras distintas y las cuatro caracteristicas de
    temporizacion se separaban un 10 %.

    Interpolar linealmente convierte ese salto de una muestra entera en un
    desplazamiento continuo de unas milesimas de muestra. No es solo cuestion de
    paridad: a 50 Hz una muestra son 20 ms sobre respiraciones de 4-5 s, y con
    dos o tres respiraciones por ventana ese redondeo se cuela entero en el
    coeficiente de variacion.
    """
    if k == 0:
        return 0.0
    x0, x1 = float(x[k - 1]), float(x[k])
    if x1 == x0:
        return float(k)
    frac = (thr - x0) / (x1 - x0)
    if frac < 0.0 or frac > 1.0:  # defensivo: el cruce no cae en este intervalo
        return float(k)
    return (k - 1) + frac


def _transiciones(x: np.ndarray) -> list[tuple[float, int]]:
    """Hysteresis comparator: (instant in samples, new state) per limb change.

    Extracted so that `segment_breaths` and `breath_bounds` share ONE
    implementation of the comparator. Two copies of a threshold rule drift
    apart silently, and the drift only shows up as a disagreement between the
    device's timing features and the reference labels -- exactly the kind of
    difference that would then be blamed on the model.

    The comparator decides on the raw sample; the instant it reports is
    interpolated between samples -- see `_instante_cruce`.
    """
    sd = float(x.std())
    if sd < EPS:
        return []
    hi, lo = 0.25 * sd, -0.25 * sd

    state = 0  # +1 inspiring, -1 expiring, 0 unknown
    transitions: list[tuple[float, int]] = []  # (instante en muestras, new_state)
    for k, v in enumerate(x):
        if state <= 0 and v > hi:
            state = 1
            transitions.append((_instante_cruce(x, k, hi), 1))
        elif state >= 0 and v < lo:
            state = -1
            transitions.append((_instante_cruce(x, k, lo), -1))
    return transitions


def segment_breaths(x: np.ndarray, fs: float = C.FS_HZ) -> tuple[list[float], list[float]]:
    """Split a band-passed channel into inspiratory / expiratory durations.

    A hysteresis comparator at +/-0.25 sigma tracks which limb of the breath we
    are on. Hysteresis (rather than a bare zero crossing) is what stops noise
    at the turning points from shattering one breath into five.

    Returns (periods_s, ie_ratios) for every complete breath in the window.
    """
    # A complete breath is insp-start -> exp-start -> next insp-start.
    periods: list[float] = []
    ratios: list[float] = []
    for _, _, t_insp, t_exp in _respiraciones(_transiciones(x), fs):
        periods.append(t_insp + t_exp)
        ratios.append(t_insp / t_exp)
    return periods, ratios


def _respiraciones(
    transitions: list[tuple[float, int]], fs: float
) -> list[tuple[float, float, float, float]]:
    """(inicio, fin, t_insp, t_exp) de cada respiracion completa, en muestras."""
    fuera: list[tuple[float, float, float, float]] = []
    for i in range(len(transitions) - 2):
        s0, d0 = transitions[i]
        s1, _ = transitions[i + 1]
        s2, _ = transitions[i + 2]
        if d0 != 1:
            continue
        t_insp = (s1 - s0) / fs
        t_exp = (s2 - s1) / fs
        if t_insp <= 0 or t_exp <= 0:
            continue
        fuera.append((s0, s2, t_insp, t_exp))
    return fuera


def breath_bounds(
    x: np.ndarray, fs: float = C.FS_HZ
) -> list[tuple[float, float, float]]:
    """Per-breath (start, end, period_s), with start/end in fractional samples.

    `segment_breaths` computes these boundaries and then throws them away,
    keeping only durations. The two-sensor reference needs the boundaries
    themselves: the thoraco-abdominal indices are defined *per breath*, and
    a breath measured on one channel has to be the same breath on the other.

    The ribcage channel is the one that segments -- it is the strong channel
    (SNR ~100 against ~1 on the axial one) -- and the same boundaries are then
    applied to the abdominal channel. Segmenting each channel independently
    would pair breath k of one with breath k of the other and silently
    mis-pair as soon as either detector drops a cycle.
    """
    return [(s0, s2, (s2 - s0) / fs)
            for s0, s2, _, _ in _respiraciones(_transiciones(x), fs)]


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    a = np.asarray(values, dtype=np.float64)
    m = float(a.mean())
    return float(a.std() / m) if abs(m) > EPS else 0.0


# --------------------------------------------------------------------------
# Stage 4: the feature vector
# --------------------------------------------------------------------------


def _channel_features(x: np.ndarray, fs: float) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(x * x)))
    crossings = int(np.count_nonzero(np.diff(np.signbit(x))))
    feats = {
        "rms": rms,
        "p2p": float(x.max() - x.min()),
        "zcr": crossings / (len(x) / fs),
    }
    feats.update(spectral_features(x))
    return feats


def extract_features(window: dict[str, np.ndarray], fs: float = C.FS_HZ) -> np.ndarray:
    """Compute the 29-element feature vector for one window.

    `window` holds the already-filtered channels sliced to WINDOW_N samples.
    Output order is fixed by `config.FEATURE_NAMES` and is load-bearing: the C
    extractor and the trained model both depend on it.
    """
    tilt, axial = window["tilt"], window["axial"]
    out: dict[str, float] = {}

    for ch_name, x in (("tilt", tilt), ("axial", axial)):
        for k, v in _channel_features(x, fs).items():
            out[f"{ch_name}_{k}"] = v

    # Cross-channel. The ratio is the primary thoracic-vs-diaphragmatic cue.
    t_rms, a_rms = out["tilt_rms"], out["axial_rms"]
    out["log_tilt_axial_ratio"] = float(np.log10((t_rms + EPS) / (a_rms + EPS)))

    tc, ac = tilt - tilt.mean(), axial - axial.mean()
    denom = float(np.sqrt((tc * tc).sum() * (ac * ac).sum())) + EPS
    out["tilt_axial_xcorr"] = float((tc * ac).sum() / denom)

    # Breath timing, measured on tilt: per the physics note in synth.py it is
    # the higher-SNR channel in every class, so segmentation is most reliable
    # there even for diaphragmatic breathing.
    periods, ratios = segment_breaths(tilt, fs)
    out["breath_rate_bpm"] = float(60.0 / np.mean(periods)) if periods else 0.0
    out["breath_period_cv"] = _cv(periods)
    out["ie_ratio_mean"] = float(np.mean(ratios)) if ratios else 0.0
    out["ie_ratio_cv"] = _cv(ratios)

    # Artifact detectors, on the raw (unfiltered) stream.
    mag, hf, gyro = window["accel_mag"], window["hf"], window["gyro_mag"]
    mag_ac = mag - mag.mean()
    out["hf_energy_ratio"] = float(
        np.sqrt(np.mean(hf * hf)) / (np.sqrt(np.mean(mag_ac * mag_ac)) + EPS)
    )
    out["jerk_max"] = float(np.abs(np.diff(mag)).max() * fs) if len(mag) > 1 else 0.0
    gyro_ac = gyro - gyro.mean()
    out["gyro_rms"] = float(np.sqrt(np.mean(gyro_ac * gyro_ac)))

    missing = set(C.FEATURE_NAMES) - set(out)
    if missing:
        raise RuntimeError(f"feature extractor is missing: {sorted(missing)}")
    return np.array([out[n] for n in C.FEATURE_NAMES], dtype=np.float32)
