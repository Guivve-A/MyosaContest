# ADR-0004 — A generated Python↔C contract, with mandatory golden vectors

- **Status:** Accepted
- **Date:** 2026-07-27
- **Decided by:** project lead and both embedded tracks

## Context

The characteristic TinyML failure is not that the device fails to boot. It is
that it boots, runs, and shows a high-confidence verdict — that is wrong, because
the DSP in C does not compute exactly the same thing as the DSP in Python the
model was trained with. That failure is silent, and in a medical feedback device
it is worse than a hard failure: nobody notices.

The sources of drift are mundane and numerous: a constant updated on one side and
not the other, a different feature order, `filtfilt` against `filt`, a biquad
coefficient transcribed by hand, a Hann window normalised differently.

## Decision

**1. `ml/pneumocoach/config.py` is the single source of truth.** If a number
appears in Python and in C, it lives there.

**2. The C artefacts are generated, never hand-written.**
`ml/scripts/emit_c_artifacts.py` produces:

| File | Contents |
|---|---|
| `firmware/include/pneumocoach_config.h` | Constants, I²C addresses, SOS coefficients, standardisation vectors, class table, Hann window |
| `firmware/include/pneumocoach_model.h` + `src/pneumocoach_model.c` | The INT8 flatbuffer as a 16-byte-aligned array |
| `firmware/include/pneumocoach_golden.h` | 12 raw int16 windows, their features according to Python, and the expected verdict |

All three carry a *GENERATED FILE — DO NOT EDIT* header.

**3. The parity test is blocking.** The firmware replays the golden windows
through the real C path and compares against `PC_GOLDEN_FEATURES` within
`PC_GOLDEN_RTOL` (2e-2 relative, 1e-4 absolute). Test fails → build fails → it
does not merge.

## Rationale

The 2e-2 tolerance is not laxity: C uses `float32` and NumPy `float64`. They will
never match bit for bit. Any deviation larger than that is not rounding, it is an
algorithmic difference — which is exactly what we want to catch.

We chose to apply the `(x−μ)/σ` standardisation **in C, outside the graph**,
rather than folding it into the first dense layer. Folding is more efficient by a
few cycles, but it hides the numbers. When somebody debugs a bad verdict at 2 in
the morning, they want to be able to print `μ` and `σ`.

## Consequences

- Changing a feature is no longer a local change: you have to touch
  `config.py`, port it to C, regenerate and retrain. **That friction is
  intentional** — it discourages adding features without thinking.
- `pneumocoach_golden.h` is ~216 KB of source (~86 KB in flash). It only enters
  the test build, not the release one. If compile time becomes annoying, reduce
  `N_GOLDEN`.
- The model array needs `__attribute__((aligned(16)))`. Without it, TFLM fails
  schema verification with a message that looks nothing like an alignment
  problem. It is already in the generator.
- Resist the temptation to "quickly fix" the C header. Any manual edit is
  destroyed by the next regeneration and, in the meantime, makes the firmware
  disagree with the model it is running.
