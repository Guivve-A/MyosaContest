# ADR-0001 — Drop the NVIDIA TAO Toolkit; use TensorFlow/Keras → TFLite INT8

- **Status:** Accepted
- **Date:** 2026-07-27
- **Decided by:** modelling track, ratified by the project lead

## Context

The reference workflow we started from (§2.1, §3.2) places the NVIDIA TAO Toolkit
as "the central pillar of the AI engineer's workflow", with pruning and QAT
through the TAO CLI and export via ONNX.

## Decision

We do not use TAO. The pipeline is Keras → `TFLiteConverter` with full-integer
quantisation (`TFLITE_BUILTINS_INT8`, `int8` input and output).

## Rationale

TAO is built for **computer vision**: classification, detection and segmentation
over images, with transfer learning from pretrained NGC models. Our problem is a
classifier over 29 tabular features derived from an accelerometry time series.

Concretely, what TAO brings does not apply:

- **There is no pretrained NGC model** for 1-D respiratory accelerometry.
  Transfer learning, which is TAO's reason to exist, has nothing to transfer
  from.
- **The export path targets TensorRT/DeepStream**, not TFLite Micro on Xtensa. We
  would have to leave through ONNX and re-enter with third-party converters,
  adding two intermediate formats and their failure modes.
- **It requires an NVIDIA GPU, Docker and an NGC account.** That is
  infrastructure the team does not have and does not need: our model trains in
  20 seconds on CPU.
- **Structured pruning has nothing to prune** in a 29→24→16→4 MLP. It is 1,349
  parameters; the quantised model weighs 4,888 bytes against a budget of 8,192.

Adopting TAO would have cost roughly a week of setup to produce a worse model by
a more fragile route.

## Consequences

- We lose TAO's automated QAT. Mitigation: post-training quantisation with a
  representative dataset measures **+0.27 pp** of accuracy change — within the
  noise, no degradation. If the drop ever exceeds 2 pp we implement QAT with
  `tensorflow_model_optimization`, which is a dependency rather than a platform.
- The paper cannot cite TAO. Nor should it: citing a tool that was not used is an
  integrity problem, not a marketing one.
- Edge Impulse is kept as an option for exploratory DSP prototyping, but it is
  not on the critical path.

## Alternatives considered

- **Edge Impulse Studio end to end.** Genuinely good for this and a valid option.
  Rejected because its DSP block generator produces C code we cannot audit line
  by line, and verifiable Python↔C parity is exactly what protects this project
  against silent failure.
- **PyTorch → ONNX → TFLite.** One more intermediate format with no benefit
  whatsoever.
