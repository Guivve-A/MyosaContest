# ADR-0003 — Tilt/axial decomposition, and deliberately overlapping rate ranges

> **PARTIALLY REVERSED on 2026-08-18 by [ADR-0006](0006-the-ratio-premise-is-false.md).**
> The ratio `log10(tilt_rms/axial_rms)` **does not separate** on real data:
> Cohen's d = −0.05 on the first subject, against ≈3.2 predicted in simulation.
> Both channels separate strongly but rise together, so the ratio cancels. The
> decision to overlap the frequency ranges **is validated**. Read ADR-0006 before
> relying on anything in this document.

- **Status:** Accepted
- **Date:** 2026-07-27
- **Decided by:** modelling track, with clinical validation pending from the mentor

## Context

With a **single** IMU on the sternum you cannot directly measure the
thorax/abdomen differential that respiratory inductance plethysmography uses
(which needs two bands). We had to find a mechanical signal that did distinguish
correct from incorrect technique with one sensor.

## Decision

**1. Split the signal into two mechanically distinct channels.**

| Channel | What it is | How it is obtained |
|---|---|---|
| `tilt` | Chest-wall rotation about the medio-lateral axis | Complementary filter (α=0.98) fusing the accelerometer angle with `gy`, then band-pass |
| `axial` | Pure antero-posterior translation | `az − sin(pitch)` to remove the gravity leak, then band-pass |

The primary discriminator is `log_tilt_axial_ratio`. Thoracic breathing
**rotates** the upper chest; diaphragmatic breathing **translates** it.

**2. Make the classes' respiratory rate ranges overlap.**

| Class | BPM | Tilt (p-p) | Translation (p-p) |
|---|---|---|---|
| diaphragmatic | 6–14 | 0.5–1.8° | 3.5–9 mm |
| thoracic | 10–24 | 1.8–4.5° | 1.5–5 mm |
| rapid shallow | 20–40 | 0.4–1.5° | 0.8–3 mm |

## Rationale

**On the decomposition.** The numbers come from arithmetic, not intuition. At
0.15 Hz a 5 mm translation produces `A(2πf)² ≈ 0.45 mg`, while a 2° rotation
produces `g·sin(2°) ≈ 35 mg`. In-band noise is ~0.4 mg (400 µg/√Hz over 0.9 Hz).
The tilt channel has SNR ~100; the axial one operates near SNR 1. **The model
leans mostly on `tilt`, and that is physics, not a defect.** The axial channel
contributes almost exclusively through the ratio.

The ratio is also **invariant to body habitus**: `Subject.gain` scales tilt and
translation together, so absolute amplitude does not generalise across subjects
but the ratio does.

**On the overlapping rates.** The first version used disjoint ranges
(6–12 / 14–22 / 25–40 BPM). The classes separated with Cohen's d ≈ 7.7 on
respiratory rate alone. That is not a technique classifier: it is a tachometer,
and it would have let through exactly the failure mode that matters most
clinically — **a patient breathing slowly but still lifting the upper chest**. By
making the rate ambiguous, the network is forced to learn the mechanical
signature.

The test `test_tilt_axial_ratio_separates_thoracic_from_diaphragmatic` pins this
property: if the ratio stops separating, the design has lost its foundation and
the suite fails.

## Consequences

- Accuracy dropped from ~92 % (disjoint ranges) to 89.7 %. **The lower figure is
  the honest one.** The higher one measured how easy our own simulation was.
- Sensor placement matters. It must sit on the **upper sternum**, flat against
  the chest. A sensor on the abdomen inverts the meaning of the ratio. This has
  to be in the demo protocol and in the paper.
- **It requires clinical validation.** The premise "thoracic = rotation-dominant
  at the upper sternum" is mechanically reasonable and consistent with the
  breathing-pattern literature, but we have not measured it on a real body. It is
  the first question for the mentor and the first measurement once the kit
  arrives. If it does not hold, this ADR is reversed and the project changes
  shape.
