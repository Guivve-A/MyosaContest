"""Sanity-check the synthetic physics and the DSP front end before training.

Run:  python scripts/smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "tools"))
import _consola  # noqa: E402,F401  UTF-8 en Windows
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pneumocoach import config as C  # noqa: E402
from pneumocoach import dataset, dsp, synth  # noqa: E402


def check_parity() -> None:
    rng = np.random.default_rng(1)
    subj = synth.make_subject(0, rng)
    sig, _ = synth.render_bout("thoracic", 30.0, subj, rng)
    counts = synth.to_raw_counts(sig, subj, rng)
    phys = synth.counts_to_physical(counts).astype(np.float64)

    fast = dsp.complementary_pitch(phys[:, :3], phys[:, 3:])
    slow = dsp.reference_complementary_pitch(phys[:, :3], phys[:, 3:])
    err = float(np.abs(fast - slow).max())
    print(f"complementary filter  vectorised vs scalar reference: max|err| = {err:.3e} deg")
    assert err < 1e-9, "vectorised complementary filter diverged from the reference loop"


def check_amplitudes() -> None:
    """Confirm the channels land where the physics note in synth.py predicts."""
    rng = np.random.default_rng(7)
    subj = synth.make_subject(0, rng)
    print(f"\n{'class':<16}{'tilt_rms(deg)':>15}{'axial_rms(mg)':>15}{'bpm':>8}{'I:E':>8}")
    for key in ("diaphragmatic", "thoracic", "rapid_shallow"):
        sig, _ = synth.render_bout(key, 180.0, subj, rng)
        counts = synth.to_raw_counts(sig, subj, rng)
        ch = dsp.channels_from_counts(counts)
        s = slice(dsp.WARMUP_N, dsp.WARMUP_N + C.WINDOW_N)
        w = {k: v[s] for k, v in ch.items()}
        f = dsp.extract_features(w)
        d = dict(zip(C.FEATURE_NAMES, f))
        print(
            f"{key:<16}{d['tilt_rms']:>15.3f}{d['axial_rms'] * 1000:>15.3f}"
            f"{d['breath_rate_bpm']:>8.1f}{d['ie_ratio_mean']:>8.2f}"
        )


def check_separability() -> None:
    t0 = time.time()
    ds = dataset.generate(n_subjects=12, seed=3, bouts=10)
    dt = time.time() - t0
    print(f"\ndataset: {len(ds)} windows from 12 subjects in {dt:.1f}s")
    print(f"  class balance: {ds.class_counts()}")

    nan = int(np.count_nonzero(~np.isfinite(ds.X)))
    print(f"  non-finite feature values: {nan}")
    assert nan == 0, "feature extractor produced NaN/Inf"

    # Per-feature one-vs-rest separability, cheapest possible read: how far
    # apart are the class means in pooled-sigma units?
    print("\n  top discriminative features (max |Cohen's d| across class pairs):")
    scores = []
    for j, name in enumerate(C.FEATURE_NAMES):
        best = 0.0
        for a in range(C.N_CLASSES):
            for b in range(a + 1, C.N_CLASSES):
                xa, xb = ds.X[ds.y == a, j], ds.X[ds.y == b, j]
                if len(xa) < 5 or len(xb) < 5:
                    continue
                pooled = np.sqrt(0.5 * (xa.var() + xb.var())) + 1e-9
                best = max(best, abs(xa.mean() - xb.mean()) / pooled)
        scores.append((best, name))
    for d_val, name in sorted(scores, reverse=True)[:10]:
        print(f"    {name:<26} d = {d_val:.2f}")


if __name__ == "__main__":
    check_parity()
    check_amplitudes()
    check_separability()
    print("\nOK")
