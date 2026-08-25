# ADR-0006 — The tilt/axial ratio does not discriminate. What replaces it

- **Status:** Accepted. Partially reverses [ADR-0003](0003-tilt-axial-decomposition-and-overlapping-rates.md)
- **Date:** 2026-08-18
- **Evidence:** `s01_protocolo_20260818_174857.csv` — 795 s, first real subject
- **Decided by:** modelling track, with the data on the table

## What ADR-0003 predicted and what the hardware measured

ADR-0003 claimed that `log10(tilt_rms / axial_rms)` would separate thoracic from
diaphragmatic breathing, and that it would be robust across subjects because a
ratio is invariant to gain. On synthetic data it gave Cohen's d ≈ 3.2.

On the first real torso:

| | diaphragmatic | thoracic | Cohen's d |
|---|---|---|---|
| `log_tilt_axial_ratio` | 2.881 | 2.876 | **−0.05** |

No separation at all. **The premise, as formulated, is false.**

## Why it failed, precisely

It is not that the sternum fails to rotate differentially. It does:

| | diaphragmatic | thoracic | rest | Cohen's d |
|---|---|---|---|---|
| `tilt_rms` (degrees) | 1.044 | 2.448 | 0.644 | **−1.84** |
| `axial_rms` (mg) | 1.340 | 3.106 | 1.192 | **−2.20** |

Both channels separate strongly. But they **rise together**, so the ratio
cancels.

The modelling error was assuming that diaphragmatic breathing would be
*translation-dominant* at the sternum. The reality is that diaphragmatic
breathing **barely moves the upper sternum in any direction**, and thoracic
breathing moves it a lot in all of them. The discriminator is not a split
between two movements: it is the magnitude of the movement.

Put that way it is obvious in hindsight. The upper sternum is precisely where
diaphragmatic breathing does *not* act; the volume is displaced lower down. That
there would be no appreciable diaphragmatic translation there was predictable,
and we did not predict it.

## What the simulation had wrong

| Feature | Real vs synthetic divergence |
|---|---|
| `tilt_p2p` | 16.1 σ |
| `tilt_rms` | 15.0 σ |
| `axial_rms` | 7.3 σ |
| `hf_energy_ratio` | 20.6 σ (rapid_shallow) |

The generator in `synth.py` does not represent the real physics. **The current
INT8 model, with its 89.7 %, is trained on wrong physics** and that figure must
not be quoted as if it said anything about the device.

## What does work

The task **is learnable**. On real data, six features exceed |d| = 0.8:

| Feature | d | Family |
|---|---|---|
| `axial_rms` | −2.20 | amplitude |
| `axial_p2p` | −1.97 | amplitude |
| `tilt_p2p` | −1.90 | amplitude |
| `tilt_rms` | −1.84 | amplitude |
| `hf_energy_ratio` | +1.79 | shape |
| `gyro_rms` | −1.62 | amplitude |

A random forest over the 29 features reaches **88.5 % ± 7.2** in 5-fold
cross-validation. **That is an optimistic upper bound**: the windows overlap by
75 %, so there is leakage between folds. It is good enough to conclude that the
signal exists, not to estimate performance.

The visual test passes too: on the filtered trace, thoracic breathing reaches
±6–7° of tilt against ±2° for diaphragmatic. They are distinguishable by eye,
which was the minimum condition the consulted expert set.

One more data point: `breath_rate_bpm` gives d = −0.27 and `ie_ratio_mean` gives
d = +0.04. The subject breathed at the same rate in both blocks and changed only
the mechanics. ADR-0003's decision to overlap the frequency ranges is validated —
the model cannot cheat via frequency because there is no frequency difference
here.

## The new problem

**Five of the six discriminating features are absolute amplitudes.** Amplitude
depends on the subject's build, on how tight the strap is, and on how hard they
work. A model trained on absolute amplitude will not generalise to another
person — which is exactly why we wanted a ratio in the first place.

The recording itself shows it: the second diaphragmatic and second thoracic
blocks have visibly smaller amplitudes than the first ones. Effort drift within
a single session, same subject, same mount.

## Decision

**1. The ratio stops being the primary discriminator.** It is kept as a feature —
it costs nothing — but no power is attributed to it.

**2. Amplitudes are normalised by the subject's resting baseline.** The protocol
already records 60 s of rest at the start and at the end; that reference is
specific to each person and each placement. Dividing by it does not change
Cohen's d *within* a subject — dividing by a constant does not alter separation —
but it makes the scale comparable **between** subjects, which is where the
current model would break.

**3. Not implemented yet.** With a single subject it is impossible to check
whether the normalisation rescues generalisation. Implementing it now would swap
one unverified assumption for another.

**4. `synth.py` is demoted to a test generator.** It is useful for exercising the
pipeline, not for training the model that gets deployed. Training now depends on
real data.

## What is needed before training again

**Three subjects minimum, ideally five.** With two you can already measure
whether rest-normalisation makes amplitudes comparable across people; with one
you can measure none of that.

The recording protocol does not change: it already captures what is needed,
including the rest blocks that now turn out to be the key piece. Each session is
thirteen minutes.

## A note on how this was found

This surfaced because the project had a hypothesis written down in falsifiable
form, a tool that checked it against a binary threshold, and real data taken
before building on top. The failure appeared in the first recording rather than
in Rotterdam.

Worth recording: ADR-0003 had said from the start that "if the ratio does not
separate on real data, the mechanical premise is wrong and the project changes
shape". It changed shape.
