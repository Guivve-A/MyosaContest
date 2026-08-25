# ADR-0005 — Plain ESP-IDF 5.5.x (not Arduino, not PlatformIO), and OpenHTF out

> **Note added 2026-08-25.** The firmware that actually ships is built with
> **Arduino-ESP32 3.3.11**, which is based on ESP-IDF 5.5.5 — the same base this
> ADR selected. The door this document deliberately left open was used. The
> reasoning below is kept because the toolchain analysis still holds and explains
> why that base was picked in the first place.

- **Status:** Accepted; the Arduino layer on top was later adopted
- **Date:** 2026-07-27
- **Decided by:** project lead and core embedded track
- **Based on:** toolchain research verified on 2026-07-27

## Context

The reference research (§2.2) recommends ESP-IDF over Arduino, and OpenHTF
(§2.3, §3.4) for software-in-the-loop testing. We had to verify the real versions
and resolve a tension: MYOSA distributes **Arduino libraries**, not IDF
components.

## Decision

**Stack: ESP-IDF v5.5.x**, installed on Windows with the ESP-IDF Installation
Manager (EIM), with these components via the Component Manager:

| Component | Version | Purpose |
|---|---|---|
| `espressif/esp-tflite-micro` | v1.3.7 | Inference engine |
| `espressif/esp-nn` | v1.1.1 | Optimised kernels (installed as a dependency) |
| `espressif/esp-dsp` | v1.8.2 | 128-point real FFT |

The SSD1306 is driven with `esp_lcd_panel_ssd1306`, which ships with ESP-IDF.
MYOSA's Arduino libraries are ported: they are thin I²C wrappers and the register
map is already documented in `Resources_docs/`.

**SITL testing: `pytest` plus `wokwi-cli`, not OpenHTF.**

## Rationale

**Why IDF 5.5.x and not 6.0.2.** The current stable release is v6.0.2, but
`esp-tflite-micro` declares `idf: ">=5.0"` with no confirmed build against 6.0,
which brings breaking changes (mbedTLS→PSA migration, build system v2). 5.5.x is
also exactly the base of Arduino-ESP32 3.3.11, so if we ever need to go back to
Arduino, the door stays open. Espressif no longer uses the LTS designation: each
release gets 30 months (12 of service plus 18 of maintenance).

**Why plain IDF and not PlatformIO.** The official
`platformio/platform-espressif32` plugin has its Arduino support **frozen at core
2.0.17 / IDF 4.4.7** — four generations behind. The community *pioarduino* fork
is up to date (Arduino 3.3.11 / IDF 5.5.5) and would be the right choice *if* we
used PlatformIO with Arduino. But the deciding factor is different:
**`esp-dsp` is ESP-IDF-first** and using it from Arduino core 3.3.x requires
manual patches. We are not going to fight the build system to get an FFT.

**Why OpenHTF is out.** OpenHTF is designed for hardware test benches in
manufacturing. It has no Wokwi integration — we would have to write it from
scratch. `wokwi-cli` already runs headless simulations in CI with YAML scenario
files (`wait-serial`, `expect-pin`, `take-screenshot`), which is exactly what we
need and already exists. Adopting OpenHTF would mean building an adapter in order
not to use the tool that does the job.

## Findings that change the design

**1. ESP-NN has NO assembly kernels for the LX6.** The Kconfig help text says it
literally: the assembly optimisations are for the ESP32-S3; for ESP32 and C3 it
uses generic C optimisations. The LX6 lacks the S3's vector ISA.

**This turns out not to matter.** The model is 3 `FULLY_CONNECTED` layers with
~1,144 MACs; estimated inference is 0.1–1 ms against a 20 ms tick. The 128-point
real FFT with `esp-dsp` is ~30 µs. **The bottleneck is not inference but the I²C
bus and the OLED flush.** `CONFIG_NN_OPTIMIZED=y` is enabled anyway because it is
free.

**2. Wokwi does not simulate BLE. At all.** It is the most serious gap in the
simulation. **Mandatory architectural implication:** the verdict is published to
an abstract *sink*, with implementations for UART, OLED and BLE. In Wokwi it is
compiled with the UART sink and `wokwi-cli` asserts on the serial output. BLE is
only validated on real hardware. If BLE is wired directly into the inference
logic, half the system has no test coverage.

**3. We need TWO custom chips, not one.**

| Module | In Wokwi | Action |
|---|---|---|
| MPU6050 | Built in, supports 0x69 with AD0→VCC | **Custom chip anyway**: the built-in one is driven by sliders and cannot replay a 600-sample waveform at 50 Hz |
| SSD1306 | Built in, 0x3C | Use the built-in |
| BMP180 | Built in | Use the built-in |
| APDS9960 | **Does not exist** | Custom chip mandatory at 0x39 |

The research only anticipated the MPU6050 chip. The custom-chip API is free
(Wokwi Club is for private projects) and `i2c_config_t` accepts an arbitrary
7-bit address, so both are viable.

**4. Wokwi CI: 50 minutes/month on the free plan** (200 hobby, 2000 pro). The
test matrix has to be budgeted. Scenarios are in **alpha** and their format may
change.

## Consequences

- Installing ESP-IDF 5.5.x with EIM is the embedded team's first unblocking
  step. ~2–3 GB.
- Porting the MYOSA Arduino libraries to IDF is real work, but bounded: they are
  register reads and writes over I²C.
- The abstract output sink is not over-engineering, it is what makes the system
  testable given that BLE cannot be simulated.
- The items left unverified (esp-tflite-micro on IDF 6.0, ESP-NN's specific
  speedup on FC layers over LX6, esp-dsp from Arduino 3.3.x) block nothing,
  because none of them is on the chosen path.
