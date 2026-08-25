# ADR-0009 — The device cannot certify that its calibration is any good

- **Status:** Accepted
- **Date:** 2026-08-19
- **Evidence:** `tools/medir_calidad_calibracion.py` over the three `s01` sessions
- **Depends on:** [ADR-0007](0007-the-problem-is-the-labels.md), [ADR-0008](0008-per-session-calibration.md)

## The problem

ADR-0008 makes every session start with two reference manoeuvres, from which the
axis `z = (x - dia) / (tor - dia)` is derived.

That axis has an obvious and silent failure mode: if the patient performs both
manoeuvres **the same way** — because they cannot tell the techniques apart,
because they misunderstood the instruction, because the sensor shifted between
one and the other — the axis is noise. And nothing downstream gives it away: the
device starts up, computes, and issues verdicts with exactly the same confident
face as if the calibration had been perfect.

We wanted the device to detect this itself and ask for a repeat. Forty seconds of
repetition is cheap; a month of sessions coached against an axis of noise is not.

## What was measured

Two candidate statistics, evaluated against a deliberately constructed null case:
**two halves of the same block**, i.e. the patient doing exactly the same thing
twice, which is precisely the failure we want to catch.

| statistic | signal (dia vs tor) | null (halves of the same block) |
|---|---|---|
| relative contrast | 0.115 – 0.571 | 0.072 – 0.379 |
| Cohen's d | 0.545 – 4.179 | 0.599 – 1.601 |

n = 12 pairs in each case, 6 windows per manoeuvre, which is what the device
collects in a real calibration.

**The two ranges overlap.** There is no threshold that accepts every good
calibration and rejects any bad one. With Cohen's d the signal's minimum (0.545)
falls even below the null's minimum (0.599).

It was checked on hardware too: with the board sitting still on a table, nobody
performing any manoeuvre at all, a complete calibration produced a contrast of
0.410 and all 29 features cleared `CONTRASTE_MINIMO`. A screen saying "29/29
usable" there would be lying.

## Decision

**The device shows the contrast as a number and uses it for nothing.**

- No traffic light, no threshold, no gating of the transition to coaching.
- The display reads `CONTRAST (NO THRESHOLD)`, not `CALIBRATION OK`.
- `CONTRASTE_MINIMO` stays, but only as what it always was: a numerical guard
  against dividing by a null axis. It is not a quality measure, and the code says
  so explicitly.
- `CONTRASTE_ALERTA` was removed, having existed for about half an hour. A
  constant that looks like a validated threshold is worse than none at all:
  somebody will reuse it assuming somebody measured it.

## Why this is the right decision and not a surrender

A threshold added "just in case" is not conservative, it is worse than nothing.
It turns a visible failure — the clinician sees an odd number and decides — into
an invisible one: the device gave its blessing, so nobody looks.

And the result is not surprising. It is
[ADR-0007](0007-the-problem-is-the-labels.md) again, from another angle: **the
drift within a manoeuvre is the same order as the difference between
manoeuvres**. While that holds there is nothing to separate, and no statistic can
invent it. That the same limit appears in two independent analyses is evidence
that the limit is real and lives in the data, not in the method.

## What would reverse this

The same thing that unblocks ADR-0007: an independent reference that labels the
manoeuvres — a second abdominal accelerometer used only for labelling, clinical
supervision, or video review. With reliable labels,
`tools/medir_calidad_calibracion.py` is run again; if the ranges separate then,
the gate goes in and this ADR is superseded.

Until then the number is displayed and logged in every session. Once there are
enough sessions with a known outcome, that log is exactly the dataset needed to
decide this with evidence.
