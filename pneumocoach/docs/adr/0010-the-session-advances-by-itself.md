# ADR-0010 — The session advances by itself; gestures go out of scope

- **Status:** Accepted
- **Date:** 2026-08-23
- **Team decision:** yes, after laying out the cost of continuing to chase it

## What was attempted

The APDS9960 is the **only input device in the kit**. Without it the session
could not advance without a PC attached, and that undid what the rest of the
firmware had achieved: a clinical session that runs on its own.

The full driver was written — no library, like the OLED and the IMU — with
decoding of all four directions, bounded FIFO reads so it does not steal samples
from the IMU, debouncing, and filtering by session state. It lives in
`firmware/arduino/pneumocoach_capture/apds.h` and it compiles.

**It never detected a gesture.**

## What was ruled out, and what was confirmed

| hypothesis | result |
|---|---|
| The module is not on the bus | False: it responds at `0x39` |
| The chip ignores writes | False: 11 of 11 configuration registers read back correct |
| Proximity gain was at minimum | **True and fixed.** `CONTROL` was never written, so it sat at 1×. Now 4×, with twice as many and longer pulses. The symptom did not change |
| Proximity offsets were flattening the reading | False: measured, they were already zero. Written anyway |
| The sensor does not measure | False: `PDATA` responds and varies, `STATUS` flags valid data |
| It is not an APDS9960 | **A false conclusion from a diagnostic of ours.** See below |

What remains unexplained: the identifier is **`0x9E`**, which is not in the
datasheet (`0xAB`, `0x9C`, `0xA8`) and which **the manufacturer's own library
would reject**. With the reads already proven correct, the value is genuine.

And one measurement is still missing: what proximity reads with a finger touching
the lens. That would separate "the infrared LED does not emit" from "it is a
matter of distance or threshold". It is a minute of work and it is still
available if anyone wants to pick it up.

## The diagnostic error, because it cost more than the symptom

A register dump was written that compared the registers against their **reset**
values and concluded *"probably not an APDS9960"*. That was false.

The sensor has its own power and no reset line: **it does not restart when the
ESP32 does.** What was read "at boot" were not reset values but the ones this
same firmware had written on the previous run. That they matched exactly what had
been written was, in fact, proof of the opposite: the chip accepts and retains
configuration.

That conclusion nearly sent someone to replace a module that works. A diagnostic
that concludes falsely is worse than none, because it redirects the work with the
authority of a measurement. The dump now compares against what was written and
separates the read-only registers, which there is no sense in judging.

## The decision

**The session advances by itself.** No button, no gesture, no PC:

```
SETTLING ──20 s──▶ READY ──10 s──▶ CAL DIAPHRAGMATIC ──30 s──▶
    CAL THORACIC ──30 s──▶ CALIBRATED ──8 s──▶ SESSION
```

The `n` and `k` serial commands still exist as a manual override for rehearsal
and for the demo, but they **are not needed** to complete a session.

The APDS9960 is no longer initialised. Turning on its gesture engine would mean
pulsing its infrared LED for nothing. The I²C scan still reports the module as
present, and `apds.h` stays in the tree, complete and unused, in case it is
revisited.

## Why this is better, not a workaround

Worth saying plainly: we arrived here because the gestures did not work and six
days remained. But the result is **clinically superior** to the original plan,
and that is not a retrospective rationalisation — it is the same argument that
justified gestures in the first place, taken to its conclusion.

The reason for wanting gestures was that the patient should touch nothing: they
are in the middle of a breathing exercise, with the sensor taped to the sternum
and **both hands on their body**, because the two-hand cue requires it. Anything
they have to operate is a movement that ends up in the channel we measure.

A gesture over the chest is *less* movement than pressing a button, but it is
still movement — and worse: during the reference manoeuvres the hands are exactly
where the sensor is looking, so gestures had to be ignored precisely in the
phases where the patient moves most. The gesture version already carried that
state filter; it was a feature that had to be disabled where it got most in the
way.

**Zero inputs solves the whole problem instead of reducing it.**

## What would reverse this

One measurement: proximity with a finger touching the lens.

- If it rises to 100–255, the chain works and only `GPENTH` needs lowering.
  Gestures would come back as an optional control, never as a requirement.
- If it stays at baseline (~4), the infrared emitter is not reaching the
  photodiode and it is a hardware fault in the module.

In neither case would the session stop advancing on its own. That part is
settled.
