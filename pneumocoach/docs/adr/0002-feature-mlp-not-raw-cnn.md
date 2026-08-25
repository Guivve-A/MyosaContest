# ADR-0002 — A classifier over DSP features, not a 1-D CNN over the raw signal

- **Status:** Accepted. Its figures come from the retracted synthetic generator — see the note below
- **Date:** 2026-07-27
- **Decided by:** modelling track

> **Note added 2026-08-25.** The architectural decision stands. The numbers in
> the results table below were measured against the synthetic generator that
> [ADR-0006](0006-the-ratio-premise-is-false.md) later showed to be wrong physics,
> and they are **retracted**. They are kept here because the reasoning that led to
> the decision is still valid and erasing the evidence would misrepresent how it
> was made. The current figure is in
> [`TECHNICAL_STATUS.md`](../TECHNICAL_STATUS.md); the deployed model is now
> 29 → 32 → 16 → 3 in 5,448 bytes, not 29 → 24 → 16 → 4.

## Context

The reference research (§2.1, §3.2) proposes 1-D CNN architectures over
accelerometry data. Our academic mentor, reviewing the EOI, explicitly asked us
**not to commit to a CNN** in the proposal and to keep the language general
("from lightweight feature-based classifiers to compact neural networks").

## Decision

A DSP front end in C producing **29 features** per 12-second window, and a
29→24→16→4 MLP with ReLU activations and a softmax output. Two baselines
(logistic regression and random forest) are trained on every run.

## Rationale

**Memory arithmetic.** A raw window is 600 samples × 6 axes = 3,600 floats =
14.4 KB of input tensor alone, before any intermediate activation. The feature
vector is 29 floats = 116 bytes. On 520 KB of SRAM shared with FreeRTOS, I²C
buffers and the OLED framebuffer, that difference decides whether the system
boots.

**The DSP is domain knowledge, not a layer that has to be learned.** The
0.1–1 Hz band-pass, the complementary filter and the tilt/translation
decomposition encode chest-wall physics that a CNN would have to rediscover from
synthetic data we generated ourselves. Learning from our own simulation what we
already know adds no information.

**Clinical interpretability.** When the mentor asks why the device said
"thoracic", the answer is `log_tilt_axial_ratio = 0.83` — the chest rotated 6.8
times more than it translated. With a CNN over the raw signal the answer is a
saliency map. In a medical context that matters.

**It keeps the generality the mentor asked for.** The paper describes the
classifier by its interface (29 features → N classes), not by its architecture.
Swapping the MLP for a GBT or a small CNN over the decimated window invalidates
nothing that is written.

## Measured results (60 simulated subjects, split by subject) — RETRACTED

| Model | Accuracy (test) | macro-F1 | Size |
|---|---|---|---|
| Logistic regression | 0.8854 | 0.8807 | — |
| Random forest (200 trees) | 0.9081 | 0.9049 | does not fit |
| MLP float32 | 0.8995 | 0.8965 | — |
| **MLP INT8 (deployed)** | **0.8968** | **0.8936** | **4,888 B** |

These numbers describe a simulation, not a chest. See the note at the top.

## Consequences

- **The random forest beats the MLP by 1.1 pp.** That has to be said in the
  paper, not hidden. It is the honest baseline and it suggests the problem is
  largely piecewise-linear over these features. A pruned GBT could fit on the
  ESP32 and is declared future work.
- If a failure mode appears that the features do not capture, the answer is to
  add a feature — not to retrain a bigger network. The front end is where the
  knowledge lives.
- Any new feature forces: updating `config.py`, porting it to C, regenerating the
  golden vectors and retraining. The cost of adding a feature is real, and that
  is deliberate.
