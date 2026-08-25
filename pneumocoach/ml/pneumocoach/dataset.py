"""Turn simulated recordings into a windowed, subject-split feature dataset.

Two methodological rules are enforced here rather than left to the caller,
because getting either wrong silently inflates accuracy:

  1. **Split by subject, never by window.** Consecutive windows overlap by 75%
     and share a mounting angle, a bias vector and a noise realisation. A
     random window split leaks all of that across the boundary and reports a
     number the demo booth will not reproduce.

  2. **A window keeps a label only if it is pure.** With a 12 s window and
     bouts that change technique, some windows straddle a transition. Those get
     dropped rather than assigned to whichever class happens to hold the
     majority, so the model is never taught that half a thoracic breath is
     diaphragmatic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from . import dsp
from .synth import Recording, render_cohort

# Fraction of a window's samples that must carry the same label for the window
# to be usable. 1.0 would discard too much; 0.9 keeps transitions honest.
PURITY = 0.90

# A window is called `artifact` if this fraction of it is contaminated. It is
# deliberately small and asymmetric to the purity rule above: a two-second
# cough inside a twelve-second window is enough to make the coaching verdict
# untrustworthy, so the device should refuse rather than average over it.
CONTAMINATION_TRIGGER = 0.08


@dataclass
class Dataset:
    X: np.ndarray  # (n, N_FEATURES) float32
    y: np.ndarray  # (n,) int64
    sid: np.ndarray  # (n,) int64 -- subject id, for grouped splits
    feature_names: tuple[str, ...] = C.FEATURE_NAMES

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(self.X[mask], self.y[mask], self.sid[mask], self.feature_names)

    def class_counts(self) -> dict[str, int]:
        return {
            c.key: int(np.count_nonzero(self.y == c.idx)) for c in C.CLASSES
        }


def windows_from_recording(rec: Recording) -> tuple[np.ndarray, np.ndarray]:
    """Filter the whole recording once, then cut overlapping windows from it."""
    chans = dsp.channels_from_counts(rec.counts, fs=rec.fs)

    feats: list[np.ndarray] = []
    labels: list[int] = []

    # `artifact` dejo de ser una clase: entrenarla junto a las tecnicas costaba
    # 5 puntos en la pregunta que importa (0.694 contra 0.750 cruzando
    # protocolos), porque una tos esta tan lejos de cualquier tecnica que la
    # frontera entre ellas se gasta en separarla.
    #
    # Sin esa clase, una ventana contaminada NO tiene etiqueta valida. Se
    # descarta, que es lo unico defendible: reetiquetarla como la tecnica
    # mayoritaria seria ensenarle al modelo que una tos es respirar.
    artifact_idx = C.CLASS_INDEX.get("artifact")
    start = dsp.WARMUP_N  # skip the high-pass settling transient
    while start + C.WINDOW_N <= len(rec.counts):
        end = start + C.WINDOW_N
        dirty = float(rec.contaminated[start:end].mean())

        if dirty >= CONTAMINATION_TRIGGER:
            label: int | None = artifact_idx   # None si la clase no existe
        elif dirty > 0.0:
            # Contaminated, but not enough to call it an artifact. Neither
            # answer is defensible, so it trains on nothing.
            label = None
        else:
            counts = np.bincount(rec.labels[start:end], minlength=C.N_CLASSES)
            top = int(counts.argmax())
            label = top if counts[top] / C.WINDOW_N >= PURITY else None

        if label is not None:
            window = {k: v[start:end] for k, v in chans.items()}
            feats.append(dsp.extract_features(window, fs=rec.fs))
            labels.append(label)

        start += C.HOP_N

    if not feats:
        return np.empty((0, C.N_FEATURES), np.float32), np.empty(0, np.int64)
    return np.stack(feats), np.asarray(labels, dtype=np.int64)


def build(recordings: list[Recording]) -> Dataset:
    xs, ys, sids = [], [], []
    for rec in recordings:
        x, y = windows_from_recording(rec)
        if len(y) == 0:
            continue
        xs.append(x)
        ys.append(y)
        sids.append(np.full(len(y), rec.sid, dtype=np.int64))
    if not xs:
        raise RuntimeError("no usable windows -- recordings too short for the warm-up?")
    return Dataset(np.concatenate(xs), np.concatenate(ys), np.concatenate(sids))


def split_by_subject(
    ds: Dataset, val_frac: float = 0.20, test_frac: float = 0.20, seed: int = 0
) -> tuple[Dataset, Dataset, Dataset]:
    """Partition subjects (not windows) into train / val / test."""
    subjects = np.unique(ds.sid)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n:
        raise ValueError(f"{n} subjects is too few for a {val_frac}/{test_frac} split")

    test_s = set(subjects[:n_test].tolist())
    val_s = set(subjects[n_test : n_test + n_val].tolist())

    in_test = np.isin(ds.sid, list(test_s))
    in_val = np.isin(ds.sid, list(val_s))
    in_train = ~(in_test | in_val)
    return ds.subset(in_train), ds.subset(in_val), ds.subset(in_test)


def generate(
    n_subjects: int = 60,
    seed: int = 0,
    bouts: int = 14,
    **kwargs,
) -> Dataset:
    """One-call cohort simulation plus feature extraction."""
    recs = render_cohort(n_subjects, seed=seed, bouts=bouts, **kwargs)
    return build(recs)
