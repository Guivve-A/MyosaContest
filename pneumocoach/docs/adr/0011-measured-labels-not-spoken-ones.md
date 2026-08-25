# ADR-0011 — The label comes from a measurement, not from an instruction

- **Status:** Accepted for the method. The empirical result is pending
- **Date:** 2026-08-23
- **Supersedes:** the block in [ADR-0007](0007-the-problem-is-the-labels.md), whose option (a) it implements
- **Evidence:** estimator self-tests against signals of known phase and ratio

## The problem it inherits

ADR-0007 left data collection blocked: the labels recorded what the subject
**believed** they were doing, not their mechanics. Two sessions half an hour
apart inverted the relationship between classes because the instruction changed.
The block was correct and stands until a reference independent of intention
exists.

This builds one.

## The decision

A second accelerometer on the abdomen — a phone's — recording simultaneously. The
label stops being a dictated category and becomes a measured quantity, per
breath:

```
phi   = phase of X_rc · conj(X_ab)   at that breath's fundamental
rc_ix = log10(D_rc / D_ab)
```

`phi` is the thoraco-abdominal phase angle, the standard index of asynchrony:
dimensionless, uncalibrated, and impossible to fake by following an instruction.
`rc_ix` is the ratio of contributions.

**The deployed device still uses a single sensor.** The abdominal one exists only
while the dataset is being built, so the MYOSA kit constraint stays intact.

## What this ADR does NOT claim

`rc_ix` **is not the clinical RC%** and is not presented as such. The RC% in the
literature is a fraction of tidal volume, and computing it would require two
sensors with the same volume-to-signal transfer function. They do not have one:
different anatomical site, different coupled mass, and at the sternum we measure
rotation while at the abdomen we measure translation.

What *is* legitimate: within one subject and one placement, `rc_ix` is a
**monotone** axis. Its zero is arbitrary; its ordering is not. And the ordering is
all the tests need. `phi` carries none of this caveat, which is why it is the
primary index.

## Method decisions, and why

**The analysis window is exactly one breath.** Resampled to N points between its
two boundaries, the fundamental falls precisely on bin 1 of the DFT: no spectral
leakage, and a single complex coefficient per channel. Hunting for a peak in the
spectrum of one breath has nowhere near the resolution for that.

**Boundaries always come from the sternal channel**, which is the strong one, and
are applied as-is to the abdominal channel. Segmenting each channel
independently would pair breath *k* of one with breath *k* of the other and
mis-pair as soon as either detector dropped a cycle.

**Acceleration is converted to displacement before the ratio.** Without that step
the ratio would depend on respiratory frequency, and `rapid_shallow` would end up
separated by a units artefact rather than by its mechanics.

**Three taps at each end, not one.** A single common event gives the offset
between the two clocks but not the drift. With six events there are four spare
degrees of freedom and the fit's residual becomes a real measure of whether the
alignment worked; with two points and two unknowns the residual would always be
zero, even with mispaired taps.

**A residual above 40 ms means the session is unusable.** A label computed on
misaligned channels is not noisy: it is false, and the bias always runs the same
way, because the offset shifts the phase in a fixed direction.

## A bug the self-test caught

The first version of the detector located each tap with `argmax`. A tap is a
short step convolved with the smoothing, and that leaves a **flat plateau**
several samples wide: on a flat plateau the maximum is decided by noise. Measured,
it gave up to **100 ms** of disagreement between the two recordings of the same
tap — more than twice the alignment limit. It was replaced by the centroid of the
area above threshold, which also gives sub-sample resolution.

Without the self-test against known offsets, that bug would have corrupted every
phase in every session, silently and systematically.

## What remains open

The empirical result. `tools/compuerta_etiquetas.py` contains the hypothesis
registered **before** any dual session was recorded, with both outcomes decided
in advance:

- **Passes:** the instruction describes the mechanics for this subject; the
  labels of the four earlier sessions hold and the two-hand protocol is validated
  against a measurement.
- **Fails:** relabel by measured `rc_ix` and retrain; what gets published becomes
  the fraction of mislabelled blocks, which is the empirical validation of
  ADR-0007.

Either way there is a result. This ADR will be updated with whichever one
arrives.

## Reproducing

```
python tools/etiqueta_objetiva.py --autoprueba
python tools/capturar_dual.py --autoprueba
python tools/compuerta_etiquetas.py --autoprueba
```

Recording protocol: [`DUAL_SENSOR_PROTOCOL.md`](../DUAL_SENSOR_PROTOCOL.md).
