# PneumoCoach — Edge-AI Breathing Technique Coach

<p align="center">
  <img src="pneumocoach/pneumocoach-cover.png" width="720"><br/>
  <em>Wearable sternal capsule (37 × 37 × 15.5 mm) built on the MYOSA platform. A single IMU on the chest, an 8-bit neural network on the ESP32, and no cloud.</em>
</p>

> Breathing retraining is standard care in pulmonary rehabilitation — but the moment the patient goes home, nobody is watching. **PneumoCoach is that missing mirror.**

---

## 🎯 Overview

Pulmonary rehabilitation teaches patients with COPD and asthma to breathe **diaphragmatically** — abdomen expanding, upper chest still, long pursed-lip exhale. The technique is taught in clinic, then practised at home, unsupervised. Technique drift is common and the patient has no way to notice it.

PneumoCoach is a wearable that watches the chest wall and reports, **every three seconds**, how the patient is breathing relative to their own calibrated reference. Everything runs on the ESP32 — no phone, no cloud, no internet required for the core loop.

The original design rested on a mechanical hypothesis that we **falsified with our own hardware**. What we built instead — and what the measurements support — is a device that calibrates against each patient at the start of every session and classifies relative to that. The full account is in [`pneumocoach/pneumocoach.md`](pneumocoach/pneumocoach.md).

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Per-session calibration** | Two 30-second reference manoeuvres; the device classifies relative to *your own* axis |
| **Fully on-device inference** | 5,448-byte INT8 neural network, 224 µs per window, TensorFlow Lite Micro |
| **Deterministic 50 Hz sampling** | FreeRTOS task pinned to core 1, measured 50.00 Hz ± 0.00% on real hardware |
| **Confidence floor** | Below 0.60 the device shows no verdict rather than risk a wrong correction |
| **Mount-angle calibration** | Two-posture Gram–Schmidt procedure; no axis alignment required |
| **Companion app (BLE)** | Next.js + React — live biofeedback, sensor detail, session progress |

---

## 🖼️ Gallery

<p align="center">
  <img src="pneumocoach/pneumocoach-architecture.png" width="720"><br/>
  <em>Complete signal chain — sensor to verdict, entirely on-device.</em>
</p>

<p align="center">
  <img src="pneumocoach/pneumocoach-sensor-placement.png" width="720"><br/>
  <em>Placement: 35 mm below the jugular notch, over the manubrium.</em>
</p>

<p align="center">
  <img src="pneumocoach/pneumocoach-resultado-real.png" width="720"><br/>
  <em>Binary accuracy on a real chest: 0.750 (p = 4.3 × 10⁻⁶), validated across four sessions with leave-one-protocol-group-out.</em>
</p>

---



## 📦 Enclosure Design

Custom 3D-printable enclosure designed in Autodesk Inventor, optimized for the MYOSA board form factor.

<p align="center">
  <img src="pneumocoach/enclosure-hero.png" width="720"><br/>
  <em>Hero view — complete enclosure assembly with lid.</em>
</p>

<p align="center">
  <img src="pneumocoach/enclosure-exploded.png" width="720"><br/>
  <em>Exploded view — box, lid, and internal layout for the MYOSA board.</em>
</p>

<table>
  <tr>
    <td align="center"><img src="pneumocoach/enclosure-box-iso.png" width="360"><br/><em>Box — isometric</em></td>
    <td align="center"><img src="pneumocoach/enclosure-lid-iso.png" width="360"><br/><em>Lid — isometric</em></td>
  </tr>
  <tr>
    <td align="center"><img src="pneumocoach/enclosure-box-inside.png" width="360"><br/><em>Box — internal cavity</em></td>
    <td align="center"><img src="pneumocoach/enclosure-corner.png" width="360"><br/><em>Corner detail</em></td>
  </tr>
  <tr>
    <td align="center"><img src="pneumocoach/enclosure-box-top.png" width="360"><br/><em>Box — top view</em></td>
    <td align="center"><img src="pneumocoach/enclosure-lid-top.png" width="360"><br/><em>Lid — top view</em></td>
  </tr>
  <tr>
    <td align="center"><img src="pneumocoach/enclosure-front.png" width="360"><br/><em>Front profile</em></td>
    <td align="center"><img src="pneumocoach/enclosure-top.png" width="360"><br/><em>Top profile</em></td>
  </tr>
</table>

> 👉 Interactive 3D showcase available in [`pneumocoach/enclosure-showcase.html`](pneumocoach/enclosure-showcase.html) — download and open locally.

---

## 📊 Validated Results

| Tested on | Binary accuracy | 95% CI | n (effective) |
|---|---|---|---|
| Long protocol | 0.717 | [0.594, 0.819] | 60 |
| Effort protocol | 0.850 | [0.651, 0.956] | 20 |
| **Pooled** | **0.750** | **[0.647, 0.835]** | **80** |
| Chance | 0.500 | — | — |

> **p = 4.3 × 10⁻⁶** — confidence interval clear of chance. Reproduce with `python ml/scripts/entrenar_real.py`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **MCU** | ESP32-WROOM-32E (dual-core Xtensa LX6, FreeRTOS) |
| **IMU** | MPU6500 on MYOSA carrier (I²C `0x69`), ±2 g / ±250 dps |
| **Peripherals** | SSD1306 OLED, APDS9960 gesture, BMP180 barometric |
| **Inference** | TensorFlow Lite for Microcontrollers (INT8, `MicroMutableOpResolver<2>`) |
| **Training** | TensorFlow / Keras, NumPy, SciPy, scikit-learn |
| **Firmware** | Arduino-ESP32 3.3.11 (ESP-IDF 5.5.5) |
| **Companion app** | Next.js 16, React 19, Tailwind CSS 4, shadcn/ui |
| **Enclosure** | Blender 5.1 (parametric, headless CLI) |
| **Testing** | pytest — 22 contract tests guarding the Python/C boundary |

---

## 📁 Repository Structure

```
pneumocoach/
├─ pneumocoach.md              ← Full project writeup (MYOSA submission)
├─ pneumocoach-cover.png       ← Cover image
├─ pneumocoach-*.png           ← Technical figures
├─ ml/
│   ├─ pneumocoach/             Config, DSP, synth, dataset, training
│   ├─ scripts/                 train.py, emit_c_artifacts.py
│   ├─ artifacts/               INT8 model (5,448 bytes), standardiser
│   └─ tests/                   22 contract tests
├─ firmware/
│   ├─ arduino/pneumocoach_capture/   Full firmware sketch + headers
│   ├─ include/                       GENERATED C headers (do not edit)
│   ├─ src/                           Model data (generated)
│   └─ respaldo/                      Factory firmware backup
├─ tools/                       Capture, calibration, analysis, enclosure
├─ companion/                   Next.js BLE companion application
└─ docs/adr/                    11 Architecture Decision Records
```

---

## 🚀 Quick Start

### 1. Flash the firmware
```bash
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/arduino/pneumocoach_capture
arduino-cli upload -p COM9 --fqbn esp32:esp32:esp32 firmware/arduino/pneumocoach_capture
```

### 2. Verify hardware
```bash
python tools/capture.py --puerto COM9 --verificar
```

### 3. Calibrate & record
```bash
python tools/orientacion.py --puerto COM9 --sujeto s01     # mount calibration
python tools/capture.py --puerto COM9 --sujeto s01 --protocolo  # labelled session
```

### 4. Train the model
```bash
python ml/scripts/train.py --subjects 60
python ml/scripts/emit_c_artifacts.py
```

### 5. Run the companion app
```bash
cd companion && npm install && npm run dev
```

---

## 📝 Full Submission

The complete MYOSA contest entry with all technical details, the falsified premise, and the validation methodology is in:

👉 **[`pneumocoach/pneumocoach.md`](pneumocoach/pneumocoach.md)**

---

## 📜 License

MIT for source code. Hardware designs under CERN-OHL-P v2.

---

## 👥 Team

**ESPOL — Escuela Superior Politécnica del Litoral**, Guayaquil, Ecuador

Electronics and Automation Engineering students, under the academic mentorship of **Prof. Briggette Briones Morante** (Faculty of Electrical and Computer Engineering).

Built on the [MYOSA platform](https://www.makesenseedutech.com/) by MakeSense EduTech.
