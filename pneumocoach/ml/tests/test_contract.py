"""Guard the invariants that the firmware silently depends on.

These are not "does the code run" tests. Each one corresponds to a way the
Python and C halves can drift apart while both continue to work perfectly on
their own -- the class of bug that produces a device which boots, runs, shows a
confident verdict, and is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from pneumocoach import config as C
from pneumocoach import dataset, dsp, synth


# --------------------------------------------------------------------------
# The shared contract
# --------------------------------------------------------------------------


def test_feature_names_unique_and_counted():
    assert len(set(C.FEATURE_NAMES)) == len(C.FEATURE_NAMES), "duplicate feature name"
    assert C.N_FEATURES == len(C.FEATURE_NAMES)


def test_class_indices_are_dense_and_ordered():
    # The C enum, the OLED string table and the model's output axis are all
    # indexed by position. A gap or a reorder silently remaps every verdict.
    assert [c.idx for c in C.CLASSES] == list(range(C.N_CLASSES))


def test_oled_labels_fit_the_display():
    # SSD1306 at size 1 is 21 characters wide; we reserve room for a prefix.
    for c in C.CLASSES:
        assert len(c.oled) <= 16, f"{c.key}: OLED label too long ({len(c.oled)})"


def test_window_framing_is_integral():
    assert C.WINDOW_N % C.DECIM == 0, "decimation must divide the window evenly"
    assert C.DEC_N <= C.NFFT, "decimated window does not fit the FFT"
    assert C.NFFT & (C.NFFT - 1) == 0, "NFFT must be a power of two for radix-2 in C"


def test_window_spans_at_least_one_slow_breath():
    slowest_bpm = min(p.rate_bpm[0] for p in synth.PROFILES.values())
    assert C.WINDOW_S >= 60.0 / slowest_bpm, "window shorter than the slowest breath"


def test_bandpass_covers_every_simulated_rate():
    for prof in synth.PROFILES.values():
        for bpm in prof.rate_bpm:
            hz = bpm / 60.0
            assert C.BP_LOW_HZ <= hz <= C.BP_HIGH_HZ, f"{bpm} BPM falls outside the band-pass"


# --------------------------------------------------------------------------
# DSP correctness
# --------------------------------------------------------------------------


def test_vectorised_complementary_filter_matches_the_scalar_reference():
    """The firmware runs the scalar loop; training runs the lfilter form."""
    rng = np.random.default_rng(0)
    subj = synth.make_subject(0, rng)
    sig, _ = synth.render_bout("thoracic", 40.0, subj, rng)
    phys = synth.counts_to_physical(synth.to_raw_counts(sig, subj, rng)).astype(np.float64)

    fast = dsp.complementary_pitch(phys[:, :3], phys[:, 3:])
    slow = dsp.reference_complementary_pitch(phys[:, :3], phys[:, 3:])
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-9)


def test_bandpass_rejects_dc_and_passes_the_breathing_band():
    from scipy import signal

    w, h = signal.sosfreqz(dsp.SOS_BANDPASS, worN=4096, fs=C.FS_HZ)
    mag = np.abs(h)

    def gain_at(f):
        return float(mag[np.argmin(np.abs(w - f))])

    assert gain_at(0.0) < 1e-3, "DC (gravity) is not being rejected"
    assert gain_at(0.25) > 0.6, "mid-band breathing is being attenuated"
    assert gain_at(5.0) < 0.05, "high-frequency motion is leaking through"


def test_counts_roundtrip_within_one_lsb():
    rng = np.random.default_rng(3)
    subj = synth.make_subject(0, rng)
    sig, _ = synth.render_bout("diaphragmatic", 20.0, subj, rng)
    counts = synth.to_raw_counts(sig, subj, rng)
    back = synth.counts_to_physical(counts)
    requant = np.empty_like(counts)
    requant[:, :3] = np.round(back[:, :3] * C.ACCEL_LSB_PER_G)
    requant[:, 3:] = np.round(back[:, 3:] * C.GYRO_LSB_PER_DPS)
    assert np.abs(requant.astype(int) - counts.astype(int)).max() <= 1


def test_features_are_finite_for_pathological_input():
    """Silence, a dead bus and a rail-clipped sensor must not produce NaN.

    All three happen on real hardware, and a NaN reaching the interpreter is an
    unrecoverable verdict rather than a wrong one.
    """
    n = C.WINDOW_N
    cases = {
        "zeros": np.zeros((n, 6), dtype=np.int16),
        "clipped": np.full((n, 6), 32767, dtype=np.int16),
        "constant": np.full((n, 6), 1234, dtype=np.int16),
    }
    for name, counts in cases.items():
        ch = dsp.channels_from_counts(counts)
        feats = dsp.extract_features({k: v[:n] for k, v in ch.items()})
        assert np.all(np.isfinite(feats)), f"{name} produced non-finite features"
        assert len(feats) == C.N_FEATURES


def test_breath_segmentation_recovers_a_known_rate():
    fs = C.FS_HZ
    for bpm in (8.0, 15.0, 30.0):
        t = np.arange(int(60 * fs)) / fs
        x = np.sin(2 * np.pi * (bpm / 60.0) * t)
        periods, _ = dsp.segment_breaths(x, fs)
        assert periods, f"no breaths detected at {bpm} BPM"
        got = 60.0 / float(np.mean(periods))
        assert abs(got - bpm) / bpm < 0.05, f"{bpm} BPM measured as {got:.1f}"


# --------------------------------------------------------------------------
# The physics claim the whole design rests on
# --------------------------------------------------------------------------


def test_synthetic_generator_is_internally_consistent():
    """El generador sintético produce lo que sus perfiles declaran.

    OJO CON EL ALCANCE. Este test NO valida la premisa mecánica: la premisa fue
    REFUTADA sobre un tórax real (Cohen's d = -0.05, ver ADR-0006). Lo único que
    comprueba es que synth.py sigue siendo coherente consigo mismo, para que un
    cambio accidental en los perfiles no pase inadvertido.

    Los datos reales divergen de este generador en 15-16 sigma sobre las
    amplitudes de inclinación. Cualquier resultado obtenido sobre datos
    sintéticos es una prueba del pipeline, no evidencia sobre el dispositivo.
    """
    rng = np.random.default_rng(11)
    j = C.FEATURE_NAMES.index("log_tilt_axial_ratio")
    vals: dict[str, list[float]] = {"diaphragmatic": [], "thoracic": []}

    for key in vals:
        for sid in range(8):
            subj = synth.make_subject(sid, rng)
            sig, _ = synth.render_bout(key, 150.0, subj, rng)
            counts = synth.to_raw_counts(sig, subj, rng)
            ch = dsp.channels_from_counts(counts)
            sl = slice(dsp.WARMUP_N, dsp.WARMUP_N + C.WINDOW_N)
            vals[key].append(float(dsp.extract_features({k: v[sl] for k, v in ch.items()})[j]))

    d = np.array(vals["diaphragmatic"])
    th = np.array(vals["thoracic"])
    pooled = np.sqrt(0.5 * (d.var() + th.var())) + 1e-9
    cohens_d = abs(d.mean() - th.mean()) / pooled
    assert th.mean() > d.mean(), (
        "el generador dejó de ser coherente con sus propios perfiles; "
        "esto NO dice nada sobre tórax reales")
    assert cohens_d > 1.0, f"synth.py perdió separación interna (d={cohens_d:.2f})"


# --------------------------------------------------------------------------
# Dataset hygiene
# --------------------------------------------------------------------------


def test_subject_split_does_not_leak():
    ds = dataset.generate(n_subjects=12, seed=5, bouts=6)
    tr, va, te = dataset.split_by_subject(ds, seed=0)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(a.sid.tolist()) & set(b.sid.tolist())), "subject appears in two splits"
    assert len(tr) + len(va) + len(te) == len(ds)


def test_every_class_is_represented():
    ds = dataset.generate(n_subjects=10, seed=2, bouts=8)
    counts = ds.class_counts()
    for key, n in counts.items():
        assert n > 0, f"class {key} never occurs -- the model cannot learn it"


def test_contaminated_windows_are_labelled_artifact_not_averaged():
    """Regression test for the bug that cost artifact recall 49%.

    Labelling a whole bout as `artifact` because it contains one 300 ms cough
    teaches the model that clean breathing is contaminated.
    """
    rng = np.random.default_rng(9)
    subj = synth.make_subject(0, rng)
    sig, mask = synth.render_bout("thoracic", 240.0, subj, rng, artifact_prob=1.0)
    assert mask.any(), "artifact_prob=1.0 produced no contamination"
    # The disturbance must be local, not smeared over the whole bout -- if it
    # were, the per-sample mask would be no better than the bout label.
    assert mask.mean() < 0.6, "contamination mask covers almost the entire bout"


def test_windows_straddling_a_technique_change_are_dropped():
    rng = np.random.default_rng(4)
    subj = synth.make_subject(0, rng)
    a, ma = synth.render_bout("diaphragmatic", 100.0, subj, rng)
    b, mb = synth.render_bout("rapid_shallow", 100.0, subj, rng)
    rec = synth.Recording(
        sid=0,
        counts=synth.to_raw_counts(np.concatenate([a, b]), subj, rng),
        labels=np.concatenate(
            [
                np.full(len(a), C.CLASS_INDEX["diaphragmatic"], np.int8),
                np.full(len(b), C.CLASS_INDEX["rapid_shallow"], np.int8),
            ]
        ),
        contaminated=np.concatenate([ma, mb]),
        fs=C.FS_HZ,
    )
    x, y = dataset.windows_from_recording(rec)
    assert len(y) > 0
    # Every surviving window must be one class or the other, never a blend.
    assert set(np.unique(y).tolist()) <= {
        C.CLASS_INDEX["diaphragmatic"],
        C.CLASS_INDEX["rapid_shallow"],
    }


def test_mount_calibration_recovers_a_rotated_module():
    """Un modulo girado sobre el pecho debe ser recuperable por calibracion.

    El primer sujeto real monto la placa a 45 grados: la gravedad se repartio
    0.738 / 0.739 g entre los ejes Y y Z. Sin compensar, `tilt` y `axial` se
    mezclan y las caracteristicas se degradan. Con la matriz de montaje
    medida vuelven a su valor.
    """
    rng = np.random.default_rng(5)
    subj = synth.make_subject(0, rng)
    sig, _ = synth.render_bout("thoracic", 160.0, subj, rng)
    counts_ideal = synth.to_raw_counts(sig, subj, rng)

    th = np.radians(45.0)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th), np.cos(th)]])
    rotada = sig.copy()
    rotada[:, :3] = sig[:, :3] @ Rx.T
    rotada[:, 3:] = sig[:, 3:] @ Rx.T
    counts_rot = synth.to_raw_counts(rotada, subj, rng)

    sl = slice(dsp.WARMUP_N, dsp.WARMUP_N + C.WINDOW_N)

    def feats(counts, mount=None):
        ch = dsp.channels_from_counts(counts, mount=mount)
        return dsp.extract_features({k: v[sl] for k, v in ch.items()})

    j = C.FEATURE_NAMES.index("log_tilt_axial_ratio")
    ideal = feats(counts_ideal)
    sin_cal = feats(counts_rot)
    con_cal = feats(counts_rot, mount=Rx.T)

    err_sin = abs(sin_cal[j] - ideal[j])
    err_con = abs(con_cal[j] - ideal[j])
    assert err_sin > 0.2, "la rotacion deberia degradar el discriminador"
    assert err_con < err_sin / 3, (
        f"la calibracion no recupero la senal: {err_con:.3f} vs {err_sin:.3f}")


def test_gyro_bias_is_subtracted_before_integration():
    """Un sesgo sin corregir camina el angulo del filtro complementario.

    El ejemplar del kit trae ~1 dps de sesgo. Integrado, eso desplaza el
    estimado de angulo un grado por segundo: mas que la propia senal
    respiratoria, que son uno a cuatro grados.
    """
    rng = np.random.default_rng(8)
    subj = synth.make_subject(0, rng)
    sig, _ = synth.render_bout("diaphragmatic", 120.0, subj, rng)
    counts = synth.to_raw_counts(sig, subj, rng)

    bias = np.array(C.GYRO_BIAS_DPS_TIPICO)
    limpio = dsp.channels_from_counts(counts)
    corregido = dsp.channels_from_counts(counts, gyro_bias_dps=bias)
    # Restar un sesgo que no estaba presente debe cambiar la estimacion de
    # pitch; si no cambia nada, el parametro se esta ignorando.
    assert not np.allclose(limpio["pitch"], corregido["pitch"], atol=1e-6)


def test_mount_must_be_three_by_three():
    counts = np.zeros((C.WINDOW_N, 6), dtype=np.int16)
    with pytest.raises(ValueError, match="3x3"):
        dsp.channels_from_counts(counts, mount=np.eye(4))


@pytest.mark.parametrize("key", ["diaphragmatic", "thoracic", "rapid_shallow"])
def test_simulated_rate_lands_in_the_requested_range(key):
    """The synthesiser must actually produce the rates it advertises."""
    rng = np.random.default_rng(hash(key) % 2**31)
    subj = synth.make_subject(0, rng)
    prof = synth.PROFILES[key]
    sig, _ = synth.render_bout(key, 240.0, subj, rng)
    counts = synth.to_raw_counts(sig, subj, rng)
    ch = dsp.channels_from_counts(counts)
    periods, _ = dsp.segment_breaths(ch["tilt"][dsp.WARMUP_N :], C.FS_HZ)
    assert periods
    bpm = 60.0 / float(np.mean(periods))
    lo, hi = prof.rate_bpm
    assert lo * 0.7 <= bpm <= hi * 1.3, f"{key}: measured {bpm:.1f} BPM, profile says {lo}-{hi}"


# --------------------------------------------------------------------------
# Calibración por sesión (ADR-0008)
# --------------------------------------------------------------------------


def test_calibracion_mapea_referencias_a_cero_y_uno():
    """El eje del paciente: su diafragmática en 0, su torácica en 1."""
    from pneumocoach.calibracion import ReferenciaSesion

    rng = np.random.default_rng(0)
    d = rng.normal(2.0, 0.1, (8, C.N_FEATURES))
    t = rng.normal(5.0, 0.1, (8, C.N_FEATURES))
    r = ReferenciaSesion.desde_ventanas(d, t)

    np.testing.assert_allclose(r.normaliza(r.dia)[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(r.normaliza(r.tor)[0], 1.0, atol=1e-9)


def test_calibracion_es_invariante_a_ganancia_y_offset():
    """Lo que la calibración tiene que arreglar: deriva afín entre sesiones.

    Si una sesión mide todo escalado por 3 y desplazado por 7, la proyección
    sobre el eje del propio paciente debe dar el mismo resultado. Ese es
    exactamente el modo de deriva que medimos entre sesiones reales.
    """
    from pneumocoach.calibracion import ReferenciaSesion

    rng = np.random.default_rng(1)
    d = rng.normal(2.0, 0.2, (8, C.N_FEATURES))
    t = rng.normal(5.0, 0.2, (8, C.N_FEATURES))
    x = rng.normal(3.5, 0.2, (5, C.N_FEATURES))

    base = ReferenciaSesion.desde_ventanas(d, t).normaliza(x)
    g, off = 3.0, 7.0
    derivada = ReferenciaSesion.desde_ventanas(d * g + off, t * g + off).normaliza(x * g + off)
    np.testing.assert_allclose(base, derivada, rtol=1e-8, atol=1e-8)


def test_calibracion_neutraliza_caracteristicas_sin_contraste():
    """Una característica igual en ambas maniobras no distingue nada.

    Dividir por ese eje degenerado amplificaría ruido hasta dominar el vector,
    así que se neutraliza en vez de propagarse.
    """
    from pneumocoach.calibracion import ReferenciaSesion

    d = np.ones((5, C.N_FEATURES)) * 2.0
    t = np.ones((5, C.N_FEATURES)) * 5.0
    t[:, 3] = 2.0  # sin contraste en la característica 3
    r = ReferenciaSesion.desde_ventanas(d, t)

    assert not r.informativas[3]
    z = r.normaliza(np.full((1, C.N_FEATURES), 9.0))
    assert z[0, 3] == 0.0
    assert np.all(np.isfinite(z))


def test_calibracion_rechaza_conjunto_de_caracteristicas_distinto():
    """Una referencia calculada con otras características no se puede reusar."""
    from pneumocoach.calibracion import ReferenciaSesion

    r = ReferenciaSesion(dia=np.zeros(C.N_FEATURES), tor=np.ones(C.N_FEATURES))
    d = r.to_dict()
    d["feature_names"] = ["otra_cosa"]
    with pytest.raises(ValueError, match="recalibrar"):
        ReferenciaSesion.from_dict(d)
