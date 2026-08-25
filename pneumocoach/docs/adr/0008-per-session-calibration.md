# ADR-0008 — Per-session calibration: the question becomes relative

- **Status:** Accepted. **Its figures are superseded** — see the note below
- **Date:** 2026-08-18
- **Evidence:** three sessions of subject `s01`, leave-one-session-out
- **Supersedes the formulation of:** [ADR-0003](0003-tilt-axial-decomposition-and-overlapping-rates.md), [ADR-0006](0006-the-ratio-premise-is-false.md)

> **Note added 2026-08-25.** The decision in this ADR stands: the device
> calibrates against the patient at the start of every session. Two things in it
> are out of date, and are kept rather than erased because that is how this
> repository handles being wrong.
>
> - **The figure.** This ADR reports 0.606, not significant. A fourth session and
>   a switch to leave-one-**protocol-group**-out give **0.750**, 95 % CI
>   [0.647, 0.835], p = 4.3 × 10⁻⁶. See
>   [`TECHNICAL_STATUS.md`](../TECHNICAL_STATUS.md).
> - **"Only one feature keeps the sign of its effect."** That was false when
>   written. Counted rather than remembered, it is 12 of 29 across four sessions
>   and 14 of 29 across the first three (`python tools/medir_cociente.py`). What
>   *is* true, and is what the argument actually needs, is that `tilt_rms` — the
>   strongest channel — changes sign between sessions while `axial_rms` does not.

## The problem, measured

Three sessions of the same subject, same mount, minutes apart:

| | cross-session accuracy |
|---|---|
| Absolute classifier, 29 features | **0.673** |
| Chance | 0.500 |

`tilt_rms` and `log_tilt_axial_ratio` change the sign of their effect between
sessions.

And that is the easiest generalisation imaginable: it does not even change
person.

## Why the model side cannot fix this

The question "is this window thoracic in absolute terms?" has no stable answer,
because the relationship between technique and sternal mechanics depends on the
subject, on the mount, and on how they execute on the day. It is not noise that
averages out with more data: the label does not determine the signal.

The question that does have a stable answer is relative: **does this look more
like YOUR thoracic or YOUR diaphragmatic, today?**

## Decision

At the start of every session the device guides two reference manoeuvres of
**30 seconds** each. From each one an average feature vector is extracted, and
those two vectors define a per-feature axis:

```
z = (x - ref_dia) / (ref_tor - ref_dia)
```

The patient's diaphragmatic breathing lands at 0 and their thoracic at 1,
whatever their build, their mount or their effort that day. The model works on
`z`.

It is a per-feature affine transform, so it corrects offset and gain at the same
time — the two ways the signal was drifting.

Implementation in `ml/pneumocoach/calibracion.py`.

## Measured result

Leave-one-session-out. The reference windows come from the first 30 s of each
manoeuvre and **are excluded from the test set**.

| Tested on | Uncalibrated | Calibrated | Informative fold? |
|---|---|---|---|
| s1 protocol | 0.519 | **0.606** | yes — the only genuinely different protocol |
| s2 effort | 0.500 | 0.750 | partly — trained on s3, same protocol |
| s3 effort | 1.000 | 1.000 | **no** — trained on s2, nearly identical |
| Average | 0.673 | 0.785 | inflated by s3 |

**The figure we report is 0.606, not 0.785.**

A later audit showed the average is inflated. s2 and s3 use the same effort
protocol and sit at a feature-space distance of 0.87 from each other, against
1.58–1.73 to s1. Training on one and testing on the other does not measure
generalisation: it measures that they resemble each other. Only the s1 fold tests
against a genuinely different session.

### And 0.606 is not significant

The windows overlap by 75 %, so of the s1 fold's 104 windows about 26 are
independent.

| | p (one-sided) | 95 % CI |
|---|---|---|
| nominal n, 104 | 0.019 | [0.505, 0.700] |
| **effective n, ~26** | **0.163** | **[0.406, 0.798]** |

With the effective n, **0.606 is not distinguishable from chance**: the interval
includes 0.500.

### What can be claimed, then

**Can:** per-session calibration improves the result in all three folds without
exception (+0.087, +0.250, +0.000). The sign of the improvement is consistent,
and that is evidence the direction is right.

**Cannot:** that the system classifies breathing technique better than chance on
a new session. With one subject and three sessions the evidence does not support
that claim.

## What is NOT reported, and why

A search over ~100 feature subsets found that
`axial_dom_freq + tilt_dom_freq` reaches 0.885. **That number is not a
performance estimate and must not be quoted.**

With three folds and a hundred candidates, finding a combination that scores high
by chance is expected, not surprising. The breakdown confirms it: that subset
gives 1.000 on s2 and s3 — which share a protocol and are therefore nearly
identical — and **0.654** on s1, the only genuinely different fold.

It is kept as a hint for when there are more subjects, not as a result.

## Clinical cost, stated

About **40 seconds** at the start of every session. That is a real cost. It is
not foreign to clinical practice — a physiotherapist also observes a patient
before correcting them — but it changes the product: PneumoCoach stops being
"put it on and go" and becomes "put it on, calibrate, then coach".

Side benefit: the calibration is visible and comprehensible in twenty seconds,
which makes it a good moment in a demonstration rather than a hidden limitation.

## Scope of what has been shown

**Shown:** that per-session calibration improves the result in all three folds
without exception. On the only informative fold, from 0.519 to 0.606.

**Not shown, and it has to be said just as clearly:** that the system classifies
better than chance on a new session. The confidence interval of 0.606 with the
effective n includes 0.500.

**Not shown:** that it generalises across people. Three sessions of one subject
cannot say anything about that. The multi-subject study is still needed, and is
still blocked by the labelling problem in
[ADR-0007](0007-the-problem-is-the-labels.md).

**A reasonable suspicion:** that across people the calibration matters *more*,
not less, because it adds anatomical variability on top of what we already see
between sessions. That is a hypothesis, not a result.

## Consequence for the documentation

The **89.7 %** figure that appeared in the blog post came from synthetic data
whose physics turned out to be wrong. It is retracted, and the corresponding
model moved to `ml/artifacts/RETIRADO/` so it cannot be flashed by accident.

It is replaced by **0.606 on the only informative fold, with calibration, not
significant at an effective n of ~26** — with those three conditions written out
every time it is quoted. A number without its scope is worse than no number.
