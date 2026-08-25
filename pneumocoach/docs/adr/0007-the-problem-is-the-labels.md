# ADR-0007 — The problem is neither the sensor nor the model: it is the labels

- **Status:** Accepted. Blocks bulk data collection
- **Date:** 2026-08-18
- **Evidence:** two sessions of the same subject, `s01`, thirty minutes apart
- **Consequence:** nobody is recruited until this is resolved

## The data

Same subject. Same mount, board never removed. Same calibration matrix. Half an
hour apart.

| | tilt (degrees) | log(tilt/axial) |
|---|---|---|
| **Session 1** diaphragmatic | 1.04 | 2.881 |
| **Session 1** thoracic | 2.45 | 2.876 |
| **Session 2** diaphragmatic | 4.69 | 2.98 |
| **Session 2** thoracic | 1.66 | 2.06 |

In session 1 thoracic breathing rotates **more** than diaphragmatic. In session 2
it rotates **less**. The mechanical relationship between the two classes
inverted.

Cohen's d on `log_tilt_axial_ratio`: **−0.05** in session 1, **−8.05** in session
2. And session 2's sign is the **opposite** of what ADR-0003 predicted.

The only variable that changed was the instructions. Session 2 asked the subject
to exaggerate.

## Update: the subject identified the mechanism

After seeing the result, the subject reported that during the `dia_fuerte` block
he had exaggerated lifting the sternum upwards. The data confirms it with room to
spare: that block shows **6.86° of sternal rotation**, the highest in the whole
session and 2.8 times his own thoracic breathing in session 1. A classifier
trained on session 1 labels it thoracic with 84 % confidence.

Lifting the sternum *is* thoracic mechanics. The sensor measured correctly; the
label lied.

This refines the diagnosis without changing the conclusion. The immediate cause
is not that the classes are inseparable, but that **the intensity instruction
masked the technique instruction**. The protocol put SOFT/EXAGGERATED in capitals
and left the technique as an adjective; the subject followed the salient
instruction. "Exaggerate the diaphragmatic breathing" is also ambiguous:
exaggerate what? He exaggerated the whole breath.

The design error is the protocol's, not the subject's.

And it reinforces the underlying decision: **the subject only caught his own
error because he was also the engineer analysing the data.** In a twenty-person
study nobody would have noticed, and twenty quietly wrong labels are worse than
none.

## What it means

It is not a sensor problem: acquisition gave 50.00 Hz both times.
It is not a calibration problem: same matrix, verified orthonormal.
It is not a feature problem: recomputed with the same code.

**It is a ground-truth problem.** The label "diaphragmatic" does not correspond
to a stable mechanics. An untrained subject asked to "breathe with the
diaphragm" produces whatever they believe that means — and it changes with the
wording of the instruction and with the effort level requested.

We were labelling the subject's *intention*, not their mechanics.

## Why the effort analysis gave a false positive

`analizar_esfuerzo.py` concluded "AMPLITUDE HOLDS" with 22 robust features and
100 % accuracy across effort levels. That result is an artefact of looking at one
session in isolation:

- 17 windows per cell, overlapping by 75 %, all from a single continuous 60 s
  block. Training on two cells and testing on the other two is not
  cross-validation; it is memorising four pieces of signal.
- Within session 2 the classes do separate cleanly. The problem is that session
  2's separation **contradicts** session 1's.

The methodological lesson: an intra-session metric cannot detect inter-session
instability. Sessions have to be compared explicitly, and now they are.

## Decision

**1. The twenty people are not recruited yet.** With this problem unresolved,
twenty subjects produce twenty unreliable label sets. It would be the most
expensive mistake available: a lot of work, a large dataset, and useless.

**2. A ground truth independent of the subject's intention is required.** Three
routes, in order of preference:

**(a) Two-sensor reference.** The thorax-abdomen differential is what defines the
technique clinically. A second accelerometer is placed on the abdomen — a phone's
will do — and records simultaneously. The label comes from the amplitude and
phase relationship between the two, which is objective. **The deployed device
still uses one sensor**; the second exists only to generate labels while building
the dataset. It is distillation from a richer sensor, and it does not violate the
kit constraint because it is not part of the product.

**(b) Clinical supervision.** A respiratory physiotherapist observes and confirms
that the subject is executing the technique. It is the highest-quality reference
and the one that makes the paper defensible, but it depends on availability.

**(c) Video with later review.** The subject is filmed and an expert labels
afterwards. Slower than (a), more accessible than (b).

**3. The order of work changes.** Before scaling to more subjects, we have to
show that two sessions of the same subject produce the same mechanics for the
same label. That is now the first gate.

## What remains valid

- Acquisition: deterministic 50.00 Hz, verified twice.
- Mount calibration: the matrix is correct and stable.
- The DSP chain: the channels separate and the signals are clean.
- The capture protocol: it records what is needed, including the rest blocks.
- The decision to overlap frequency ranges: still correct.

What is not valid is any claim about classification accuracy.

## A note on method

This failure appeared because a second session was recorded thirty minutes after
the first and the two were compared explicitly. With a single session, either one
would have looked conclusive — and they point in opposite directions.

Worth recording: the second session was recorded to answer a different question
(whether effort can masquerade as technique). That question went unanswered,
because a larger problem surfaced underneath it.

> **Follow-up:** the two-sensor reference of option (a) is built and validated in
> [ADR-0011](0011-measured-labels-not-spoken-ones.md).
