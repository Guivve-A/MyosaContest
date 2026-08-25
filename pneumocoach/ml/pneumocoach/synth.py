"""Physics-based synthetic MPU6050 recordings of chest-wall breathing.

Why this exists
---------------
The MYOSA kit has not arrived yet, so the model has to be bootstrapped against
simulated physics. This module does NOT emit "a sine plus noise". It models the
two mechanically distinct things a sensor taped to the upper sternum actually
sees, then renders them through the MPU6050's real transfer function:

  1. **Rotation** of the chest wall about the medio-lateral axis, theta(t).
     Gravity leaks into the antero-posterior axis as g*sin(theta), and the
     gyroscope sees d(theta)/dt directly. Thoracic breathing is rotation-heavy.

  2. **Antero-posterior translation** of the chest wall, d(t). The
     accelerometer sees its second derivative, d''(t). Diaphragmatic breathing
     is translation-heavy at the sternum.

That decomposition is the whole ballgame: the tilt/axial energy ratio is what
separates a correct diaphragmatic breath from an incorrect thoracic one, and it
survives the sensor being mounted a few degrees off.

A reality check worth writing down. At 0.15 Hz a 5 mm translation produces
A*(2*pi*f)^2 ~= 0.45 mg, while a 2 deg rotation produces g*sin(2 deg) ~= 35 mg.
The rotation channel is roughly two orders of magnitude stronger. In-band accel
noise is ~0.4 mg (400 ug/rtHz over a 0.9 Hz band), so the axial channel sits
near SNR 1 while the tilt channel sits near SNR 100. The model is expected to
lean on tilt; axial contributes mainly through the ratio. Anyone tuning this
later should not be surprised by that asymmetry -- it is physics, not a bug.

Output is raw int16 MPU6050 counts, exactly what the I2C burst read at register
0x3B returns, so the same arrays can be replayed byte-for-byte by the Wokwi
custom chip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C

G_MS2 = 9.80665


# --------------------------------------------------------------------------
# Per-class mechanical profiles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BreathProfile:
    """Ranges are (low, high); each simulated bout draws uniformly."""

    key: str
    rate_bpm: tuple[float, float]
    rate_cv: tuple[float, float]  # breath-to-breath period variability
    tilt_deg: tuple[float, float]  # peak-to-peak chest-wall rotation
    trans_mm: tuple[float, float]  # peak-to-peak AP translation
    ie_ratio: tuple[float, float]  # inspiration / expiration duration
    ie_cv: tuple[float, float]
    amp_cv: tuple[float, float]  # breath-to-breath amplitude variability


# Note on the rate ranges below: they OVERLAP on purpose.
#
# A first pass used disjoint rates (6-12 / 14-22 / 25-40 BPM) and the classes
# came apart at Cohen's d ~ 7.7 on breath rate alone. That model would not be a
# technique classifier, it would be a tachometer -- and it would wave through
# the failure mode that matters most clinically: a patient breathing slowly but
# still heaving the upper chest. Rate is deliberately made ambiguous so the
# network has to learn the mechanical signature (tilt-vs-axial energy) instead.
#
# Body habitus enters as `Subject.gain`, which scales tilt and translation
# together. Absolute amplitude is therefore NOT a reliable cue across subjects,
# but their ratio is gain-invariant -- which is exactly the feature we want the
# model to rely on.
PROFILES: dict[str, BreathProfile] = {
    # Coached target: deep, long pursed-lip exhale, regular, and critically --
    # little upper-chest rotation relative to antero-posterior excursion.
    "diaphragmatic": BreathProfile(
        key="diaphragmatic",
        rate_bpm=(6.0, 14.0),
        rate_cv=(0.03, 0.12),
        tilt_deg=(0.5, 1.8),
        trans_mm=(3.5, 9.0),
        ie_ratio=(0.30, 0.70),  # 1:3.3 .. 1:1.4
        ie_cv=(0.05, 0.18),
        amp_cv=(0.05, 0.14),
    ),
    # The habit we are trying to break: upper chest heaving, shallow, ~1:1.
    # Overlaps diaphragmatic in rate (10-14) and rapid_shallow in rate (20-24).
    "thoracic": BreathProfile(
        key="thoracic",
        rate_bpm=(10.0, 24.0),
        rate_cv=(0.08, 0.22),
        tilt_deg=(1.8, 4.5),
        trans_mm=(1.5, 5.0),
        ie_ratio=(0.60, 1.15),
        ie_cv=(0.10, 0.28),
        amp_cv=(0.10, 0.24),
    ),
    # Tachypnoea / distress: fast, small, irregular.
    "rapid_shallow": BreathProfile(
        key="rapid_shallow",
        rate_bpm=(20.0, 40.0),
        rate_cv=(0.15, 0.35),
        tilt_deg=(0.4, 1.5),
        trans_mm=(0.8, 3.0),
        ie_ratio=(0.80, 1.30),
        ie_cv=(0.20, 0.40),
        amp_cv=(0.18, 0.35),
    ),
}

# The artifact class is not a breathing pattern -- it is any of the above with
# a mechanical disturbance on top. Modelled separately in `_inject_artifacts`.
ARTIFACT_KINDS = ("cough", "motion", "speech", "sensor_fault")


@dataclass(frozen=True)
class Subject:
    """Per-subject nuisance parameters, fixed for a whole recording.

    These are what make train/test splits meaningful: a model that only works
    because it memorised one mounting angle is useless in the demo booth.
    """

    sid: int
    mount_roll_deg: float  # sensor rotated on the chest
    mount_pitch_deg: float
    posture_pitch_deg: float  # seated upright vs reclined
    posture_drift_deg: float  # slow postural sway over the recording
    gain: float  # chest compliance / body habitus scaling
    accel_bias_g: np.ndarray  # per-axis fixed offset
    gyro_bias_dps: np.ndarray


def make_subject(sid: int, rng: np.random.Generator) -> Subject:
    return Subject(
        sid=sid,
        mount_roll_deg=float(rng.normal(0.0, 6.0)),
        mount_pitch_deg=float(rng.normal(0.0, 6.0)),
        posture_pitch_deg=float(rng.uniform(-15.0, 25.0)),
        posture_drift_deg=float(abs(rng.normal(0.0, 1.2))),
        gain=float(rng.uniform(0.7, 1.4)),
        accel_bias_g=rng.normal(0.0, 0.02, 3),
        gyro_bias_dps=rng.normal(0.0, 1.5, 3),
    )


# --------------------------------------------------------------------------
# Breath timing
# --------------------------------------------------------------------------


def _breath_timeline(
    n: int,
    fs: float,
    rate_bpm: float,
    rate_cv: float,
    ie_ratio: float,
    ie_cv: float,
    amp_cv: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a per-sample excursion waveform with realistic breath asymmetry.

    Returns (excursion, phase) where excursion rises 0->1 during inspiration
    and falls 1->0 during expiration. Real breathing is not sinusoidal: the
    inspiratory limb is shorter than the expiratory one, dramatically so under
    pursed-lip coaching. We get that by warping the phase of a raised cosine
    rather than by summing harmonics, which keeps the spectrum clean.
    """
    mean_period = 60.0 / rate_bpm
    excursion = np.zeros(n, dtype=np.float64)
    phase = np.zeros(n, dtype=np.float64)

    t = 0  # sample cursor
    while t < n:
        period = max(0.6, rng.normal(mean_period, mean_period * rate_cv))
        ratio = max(0.15, min(3.0, rng.normal(ie_ratio, ie_ratio * ie_cv)))
        amp = max(0.15, rng.normal(1.0, amp_cv))

        # ratio = t_insp / t_exp, so the inspiratory fraction of the cycle is:
        fi = ratio / (1.0 + ratio)
        n_cyc = int(round(period * fs))
        if n_cyc < 2:
            break
        end = min(t + n_cyc, n)
        k = np.arange(end - t, dtype=np.float64) / n_cyc  # phase in [0,1)

        # Phase warp: compress the first `fi` of the cycle into the first half
        # of the cosine, stretch the rest into the second half.
        warped = np.where(k < fi, 0.5 * k / fi, 0.5 + 0.5 * (k - fi) / (1.0 - fi))
        excursion[t:end] = amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * warped))
        phase[t:end] = k
        t = end

    return excursion, phase


def _rotation_matrix(roll_deg: float, pitch_deg: float) -> np.ndarray:
    r, p = np.deg2rad(roll_deg), np.deg2rad(pitch_deg)
    cr, sr, cp, sp = np.cos(r), np.sin(r), np.cos(p), np.sin(p)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return ry @ rx


# --------------------------------------------------------------------------
# Core renderer
# --------------------------------------------------------------------------


def render_bout(
    cls_key: str,
    duration_s: float,
    subject: Subject,
    rng: np.random.Generator,
    fs: float = C.FS_HZ,
    artifact_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one bout as float physical units, plus a contamination mask.

    Returns `(sig, contaminated)`:
      sig          (n, 6) -- [ax, ay, az] in g, [gx, gy, gz] in dps
      contaminated (n,)   -- bool, True where a disturbance corrupts the signal

    Sensor frame convention (board flat on the sternum, subject upright):
      +X superior (towards the head)   -> reads +1 g at rest
      +Y to the subject's left
      +Z anterior (out of the chest)   -> carries breathing

    Artifacts are injected into a bout of ordinary breathing rather than being
    a breathing "class" of their own. A cough lasts 300 ms; labelling a whole
    40 s bout around it as `artifact` teaches the model that clean breathing is
    contaminated, which is how the first version of this file scored 49% recall
    on that class. The mask is what carries the truth.
    """
    n = int(round(duration_s * fs))
    if n <= 0:
        return np.zeros((0, 6)), np.zeros(0, dtype=bool)

    base_key = cls_key if cls_key in PROFILES else str(rng.choice(list(PROFILES)))
    prof = PROFILES[base_key]

    rate = rng.uniform(*prof.rate_bpm)
    excursion, _ = _breath_timeline(
        n,
        fs,
        rate,
        rng.uniform(*prof.rate_cv),
        rng.uniform(*prof.ie_ratio),
        rng.uniform(*prof.ie_cv),
        rng.uniform(*prof.amp_cv),
        rng,
    )
    # Centre it so the band-pass has nothing to chew on later.
    excursion = excursion - excursion.mean()

    # --- Mechanical channels -------------------------------------------------
    tilt_pp_deg = rng.uniform(*prof.tilt_deg) * subject.gain
    trans_pp_m = rng.uniform(*prof.trans_mm) * 1e-3 * subject.gain

    theta_deg = tilt_pp_deg * excursion  # chest-wall rotation
    theta_rad = np.deg2rad(theta_deg)
    d_m = trans_pp_m * excursion  # AP translation

    # Slow postural sway: a real person does not hold perfectly still.
    t_s = np.arange(n) / fs
    sway = subject.posture_drift_deg * np.sin(
        2.0 * np.pi * rng.uniform(0.005, 0.03) * t_s + rng.uniform(0, 2 * np.pi)
    )

    # Gyro sees the angular rate of rotation + sway, in dps.
    gy = np.gradient(theta_deg + sway, 1.0 / fs)
    # Translation acceleration, m/s^2 -> g.
    a_trans_g = np.gradient(np.gradient(d_m, 1.0 / fs), 1.0 / fs) / G_MS2

    # --- Project gravity through the instantaneous orientation ---------------
    total_pitch = np.deg2rad(subject.posture_pitch_deg) + theta_rad + np.deg2rad(sway)
    ax = np.cos(total_pitch)
    az = np.sin(total_pitch) + a_trans_g
    ay = np.zeros(n)

    # Small lateral coupling -- the chest is not a perfect hinge.
    lateral = 0.15 * rng.uniform(0.3, 1.0)
    ay += lateral * np.deg2rad(theta_deg) * rng.uniform(-1, 1)

    gx = 0.10 * gy * rng.uniform(-1, 1)
    gz = 0.08 * gy * rng.uniform(-1, 1)

    accel = np.stack([ax, ay, az], axis=1)
    gyro = np.stack([gx, gy, gz], axis=1)

    # --- Sensor mounting misalignment ---------------------------------------
    rot = _rotation_matrix(subject.mount_roll_deg, subject.mount_pitch_deg)
    accel = accel @ rot.T
    gyro = gyro @ rot.T

    out = np.concatenate([accel, gyro], axis=1)
    contaminated = np.zeros(n, dtype=bool)

    if rng.random() < artifact_prob:
        out, contaminated = _inject_artifacts(out, fs, rng)

    return out, contaminated


# The band-pass rings for a while after a transient, so the samples adjacent to
# a disturbance are corrupted too even though nothing was added to them.
GUARD_BEFORE_S = 1.0
GUARD_AFTER_S = 3.0


def _inject_artifacts(
    sig: np.ndarray, fs: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Superimpose a mechanical disturbance and report which samples it touched.

    The point of the artifact class is refusal, not diagnosis: when the patient
    coughs, talks or shifts in the chair, the coaching verdict must be
    suppressed rather than silently wrong.
    """
    sig = sig.copy()
    n = len(sig)
    contaminated = np.zeros(n, dtype=bool)
    kind = str(rng.choice(ARTIFACT_KINDS))

    def mark(start: int, dur: int) -> None:
        lo = max(0, start - int(GUARD_BEFORE_S * fs))
        hi = min(n, start + dur + int(GUARD_AFTER_S * fs))
        contaminated[lo:hi] = True

    if kind == "cough":
        # Sharp, broadband, 0.2-0.5 s, large on both accel and gyro.
        for _ in range(rng.integers(1, 4)):
            dur = int(rng.uniform(0.2, 0.5) * fs)
            start = int(rng.integers(0, max(1, n - dur)))
            env = np.hanning(dur)
            burst = rng.normal(0, 1, (dur, 6)) * env[:, None]
            burst[:, :3] *= rng.uniform(0.25, 0.9)  # g
            burst[:, 3:] *= rng.uniform(60.0, 200.0)  # dps
            sig[start : start + dur] += burst
            mark(start, dur)

    elif kind == "motion":
        # Postural shift: a large, slow, permanent-ish tilt change.
        dur = int(rng.uniform(1.0, 3.0) * fs)
        start = int(rng.integers(0, max(1, n - dur)))
        step = np.deg2rad(rng.uniform(8.0, 30.0)) * rng.choice([-1, 1])
        ramp = np.zeros(n)
        ramp[start : start + dur] = np.linspace(0, step, dur)
        ramp[start + dur :] = step
        sig[:, 0] += np.cos(ramp) - 1.0
        sig[:, 2] += np.sin(ramp)
        sig[start : start + dur, 4] += np.gradient(np.rad2deg(ramp[start : start + dur]), 1.0 / fs)
        # A postural step shifts the DC level, and the 0.1 Hz high-pass takes
        # roughly a breath cycle to walk it back out, so the corruption runs
        # well past the ramp itself.
        mark(start, dur + int(5.0 * fs))

    elif kind == "speech":
        # Talking modulates the chest wall at 2-8 Hz, well above the breathing
        # band -- this is exactly what the HF energy feature is looking for.
        dur = int(rng.uniform(2.0, 6.0) * fs)
        start = int(rng.integers(0, max(1, n - dur)))
        t = np.arange(dur) / fs
        env = np.hanning(dur)
        for f0 in rng.uniform(2.0, 8.0, 3):
            sig[start : start + dur, 2] += (
                rng.uniform(0.01, 0.04) * env * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
            )
            sig[start : start + dur, 4] += (
                rng.uniform(2.0, 8.0) * env * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
            )
        mark(start, dur)

    else:  # sensor_fault -- clipping or a dead bus returning zeros
        dur = int(rng.uniform(0.5, 2.5) * fs)
        start = int(rng.integers(0, max(1, n - dur)))
        if rng.random() < 0.5:
            sig[start : start + dur, :3] = np.sign(sig[start : start + dur, :3]) * C.ACCEL_FS_G
        else:
            sig[start : start + dur] = 0.0
        mark(start, dur)

    return sig, contaminated


# --------------------------------------------------------------------------
# Sensor model: noise, bias, quantisation
# --------------------------------------------------------------------------


def to_raw_counts(
    sig: np.ndarray, subject: Subject, rng: np.random.Generator, fs: float = C.FS_HZ
) -> np.ndarray:
    """Apply the MPU6050 noise model and quantise to int16 register values.

    Noise RMS = density * sqrt(bandwidth), with bandwidth = fs/2 for the
    sampled stream. Datasheet figures live in config.py.
    """
    sig = sig.copy()
    bw = fs / 2.0
    accel_noise_g = (C.ACCEL_NOISE_DENSITY_UG_RTHZ * 1e-6) * np.sqrt(bw)
    gyro_noise_dps = C.GYRO_NOISE_DENSITY_DPS_RTHZ * np.sqrt(bw)

    sig[:, :3] += subject.accel_bias_g + rng.normal(0.0, accel_noise_g, sig[:, :3].shape)
    sig[:, 3:] += subject.gyro_bias_dps + rng.normal(0.0, gyro_noise_dps, sig[:, 3:].shape)

    counts = np.empty(sig.shape, dtype=np.int16)
    counts[:, :3] = np.clip(
        np.round(sig[:, :3] * C.ACCEL_LSB_PER_G), -32768, 32767
    ).astype(np.int16)
    counts[:, 3:] = np.clip(
        np.round(sig[:, 3:] * C.GYRO_LSB_PER_DPS), -32768, 32767
    ).astype(np.int16)
    return counts


def counts_to_physical(counts: np.ndarray) -> np.ndarray:
    """Inverse of `to_raw_counts` scaling -- what the firmware does on-device."""
    out = counts.astype(np.float32)
    out[:, :3] /= C.ACCEL_LSB_PER_G
    out[:, 3:] /= C.GYRO_LSB_PER_DPS
    return out


# --------------------------------------------------------------------------
# Recording assembly
# --------------------------------------------------------------------------


@dataclass
class Recording:
    """One simulated session: raw counts plus two per-sample truth tracks."""

    sid: int
    counts: np.ndarray  # (n, 6) int16
    labels: np.ndarray  # (n,) int8 -- the breathing class being performed
    contaminated: np.ndarray  # (n,) bool -- a disturbance corrupts this sample
    fs: float


def render_recording(
    subject: Subject,
    rng: np.random.Generator,
    bouts: int = 12,
    bout_s: tuple[float, float] = (25.0, 60.0),
    artifact_rate: float = 0.35,
    fs: float = C.FS_HZ,
) -> Recording:
    """Chain bouts into one continuous recording for a subject.

    Bouts are contiguous so that windowing sees real transitions between
    techniques, which is what happens when a patient is being coached.
    `artifact_rate` is the per-bout probability of a disturbance somewhere
    inside it -- people cough and shift regardless of how they are breathing,
    so it applies to every technique rather than forming a bout type.
    """
    keys = [c.key for c in C.CLASSES if c.key != "artifact"]
    segs: list[np.ndarray] = []
    labs: list[np.ndarray] = []
    cont: list[np.ndarray] = []

    for _ in range(bouts):
        key = str(rng.choice(keys))
        dur = float(rng.uniform(*bout_s))
        seg, mask = render_bout(key, dur, subject, rng, fs, artifact_prob=artifact_rate)
        if len(seg) == 0:
            continue
        segs.append(seg)
        labs.append(np.full(len(seg), C.CLASS_INDEX[key], dtype=np.int8))
        cont.append(mask)

    sig = np.concatenate(segs, axis=0)
    counts = to_raw_counts(sig, subject, rng, fs)
    return Recording(
        sid=subject.sid,
        counts=counts,
        labels=np.concatenate(labs),
        contaminated=np.concatenate(cont),
        fs=fs,
    )


def render_cohort(
    n_subjects: int, seed: int = 0, **kwargs
) -> list[Recording]:
    """Render a cohort. Split downstream BY SUBJECT, never by window."""
    rng = np.random.default_rng(seed)
    out = []
    for sid in range(n_subjects):
        subj = make_subject(sid, rng)
        out.append(render_recording(subj, rng, **kwargs))
    return out
