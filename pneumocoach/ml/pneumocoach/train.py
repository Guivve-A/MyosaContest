"""Train the PneumoCoach classifier and quantise it for the ESP32.

Design notes that matter for the embedded target:

  * Feature standardisation is NOT a Keras layer. The mean/scale vectors are
    exported to `pneumocoach_config.h` and applied in C before the tensor is
    handed to the interpreter. Keeping it outside the graph means the same
    numbers are visible to the firmware author, to the golden-vector test and
    to anyone debugging a bad verdict at 2 a.m.

  * Two baselines (logistic regression, random forest) are trained alongside
    the MLP. If a linear model matches the network, the network is not earning
    its 8 KB and we should say so in the paper rather than ship it.

  * The reported headline number is INT8 accuracy on held-out SUBJECTS, not
    float accuracy on held-out windows. Everything else is diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config as C
from .dataset import Dataset, split_by_subject


@dataclass
class Standardiser:
    """z = (x - mean) / scale, with a floor on scale for constant features."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardiser":
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        # A feature that never varies in training would otherwise divide by
        # ~0 and turn a harmless constant into an infinity on device.
        scale[scale < 1e-6] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: list[list[int]]

    def summary(self) -> str:
        rows = [f"  accuracy {self.accuracy:.4f}   macro-F1 {self.macro_f1:.4f}"]
        for k, v in self.per_class_f1.items():
            rows.append(f"    F1 {k:<16} {v:.4f}")
        return "\n".join(rows)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    from sklearn.metrics import confusion_matrix, f1_score

    labels = list(range(C.N_CLASSES))
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return Metrics(
        accuracy=float((y_true == y_pred).mean()),
        macro_f1=float(f1s.mean()),
        per_class_f1={c.key: float(f1s[c.idx]) for c in C.CLASSES},
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )


def print_confusion(m: Metrics) -> None:
    keys = [c.key[:12] for c in C.CLASSES]
    header = "true \\ pred"
    print(f"    {header:<14}" + "".join(f"{k:>14}" for k in keys))
    for i, row in enumerate(m.confusion):
        print(f"    {keys[i]:<14}" + "".join(f"{v:>14}" for v in row))


# --------------------------------------------------------------------------
# Baselines -- the honesty check on whether the MLP is worth its flash
# --------------------------------------------------------------------------


def train_baselines(tr: Dataset, te: Dataset, std: Standardiser) -> dict[str, Metrics]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    xtr, xte = std(tr.X), std(te.X)
    out: dict[str, Metrics] = {}

    # sklearn >=1.7 dropped `multi_class`; multinomial is the default now.
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(xtr, tr.y)
    out["logreg"] = evaluate(te.y, lr.predict(xte))

    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1
    )
    rf.fit(xtr, tr.y)
    out["random_forest"] = evaluate(te.y, rf.predict(xte))
    return out


# --------------------------------------------------------------------------
# The deployed model
# --------------------------------------------------------------------------


def build_mlp(n_features: int, hidden=C.MLP_HIDDEN):
    import tensorflow as tf

    layers = [tf.keras.layers.Input(shape=(n_features,), name="features")]
    for i, units in enumerate(hidden):
        layers.append(tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}"))
    layers.append(tf.keras.layers.Dense(C.N_CLASSES, activation="softmax", name="verdict"))
    return tf.keras.Sequential(layers, name="pneumocoach")


def train_mlp(tr: Dataset, va: Dataset, std: Standardiser, epochs: int = 200, seed: int = 0):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = build_mlp(tr.X.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    counts = np.bincount(tr.y, minlength=C.N_CLASSES).astype(np.float64)
    weights = {i: float(len(tr.y) / (C.N_CLASSES * max(c, 1))) for i, c in enumerate(counts)}

    model.fit(
        std(tr.X),
        tr.y,
        validation_data=(std(va.X), va.y),
        epochs=epochs,
        batch_size=64,
        class_weight=weights,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=25, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
        ],
    )
    return model


# --------------------------------------------------------------------------
# INT8 quantisation
# --------------------------------------------------------------------------


def quantise_int8(model, x_rep: np.ndarray, outdir: Path) -> bytes:
    """Full-integer quantisation, INT8 in and INT8 out.

    `inference_input_type=int8` matters: it lets the firmware hand the
    interpreter a plain int8 buffer instead of paying for a float->int8
    conversion op on every window.
    """
    import tensorflow as tf

    saved = outdir / "saved_model"
    model.export(str(saved))

    def representative():
        # A few hundred real feature vectors is plenty to pin the activation
        # ranges; more just slows calibration down.
        idx = np.random.default_rng(0).choice(len(x_rep), min(500, len(x_rep)), replace=False)
        for row in x_rep[idx]:
            yield [row.reshape(1, -1).astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def tflite_predict(tflite_bytes: bytes, x: np.ndarray) -> np.ndarray:
    """Run the quantised model exactly as the device will, including the
    int8 input/output scaling. This is the number that goes in the paper."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    preds = np.empty(len(x), dtype=np.int64)
    for i, row in enumerate(x):
        q = np.clip(np.round(row / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q.reshape(1, -1))
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0].astype(np.float32)
        preds[i] = int(np.argmax((raw - out_zp) * out_scale))
    return preds


def tflite_probabilities(tflite_bytes: bytes, x: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    probs = np.empty((len(x), C.N_CLASSES), dtype=np.float32)
    for i, row in enumerate(x):
        q = np.clip(np.round(row / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q.reshape(1, -1))
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0].astype(np.float32)
        probs[i] = (raw - out_zp) * out_scale
    return probs


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run(ds: Dataset, outdir: Path, seed: int = 0) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    tr, va, te = split_by_subject(ds, seed=seed)
    print(
        f"split by subject -> train {len(tr)} / val {len(va)} / test {len(te)} windows "
        f"({len(np.unique(tr.sid))}/{len(np.unique(va.sid))}/{len(np.unique(te.sid))} subjects)"
    )

    std = Standardiser.fit(tr.X)

    print("\nbaselines (test set):")
    baselines = train_baselines(tr, te, std)
    for name, m in baselines.items():
        print(f"  {name}:")
        print(m.summary())

    print("\ntraining MLP ...")
    model = train_mlp(tr, va, std, seed=seed)
    float_pred = np.argmax(model.predict(std(te.X), verbose=0), axis=1)
    m_float = evaluate(te.y, float_pred)
    print("  float32 (test set):")
    print(m_float.summary())

    print("\nquantising to INT8 ...")
    blob = quantise_int8(model, std(tr.X), outdir)
    (outdir / "model_int8.tflite").write_bytes(blob)
    int8_pred = tflite_predict(blob, std(te.X))
    m_int8 = evaluate(te.y, int8_pred)
    print(f"  model size: {len(blob)} bytes (budget {C.MAX_MODEL_BYTES})")
    print("  INT8 (test set)  <-- headline number:")
    print(m_int8.summary())
    print()
    print_confusion(m_int8)

    drop = m_float.accuracy - m_int8.accuracy
    print(f"\n  quantisation accuracy drop: {drop * 100:+.2f} pp")

    report = {
        "n_windows": {"train": len(tr), "val": len(va), "test": len(te)},
        "n_subjects": {
            "train": int(len(np.unique(tr.sid))),
            "val": int(len(np.unique(va.sid))),
            "test": int(len(np.unique(te.sid))),
        },
        "class_counts": ds.class_counts(),
        "baselines": {k: asdict(v) for k, v in baselines.items()},
        "mlp_float32": asdict(m_float),
        "mlp_int8": asdict(m_int8),
        "model_bytes": len(blob),
        "quantisation_drop_pp": drop * 100,
        "standardiser": {"mean": std.mean.tolist(), "scale": std.scale.tolist()},
        "feature_names": list(C.FEATURE_NAMES),
        "class_keys": list(C.CLASS_KEYS),
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    np.savez(
        outdir / "standardiser.npz", mean=std.mean, scale=std.scale
    )
    return report
