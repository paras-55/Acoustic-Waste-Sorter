# Synthetic Acoustic Waste Sorter Dataset (2 classes)

## Intended use
This dataset is for prototyping audio-processing and ML code for the Sound-Based Waste Classification project. It contains only two classes: `plastic` and `metal`, matching the first prototype scope described in the hardware handover. It is synthetic audio, not physical recordings from the user's INMP441 microphone or 18 x 18 cm impact plate.

## Dataset contents
- 900 synthetic WAV clips total.
- Classes: `plastic` and `metal`.
- Training: 300 clips/class (600 total).
- Validation: 75 clips/class (150 total).
- Test: 75 clips/class (150 total).
- Duration: 2.0 seconds per clip.
- Sample rate: 16,000 Hz (16 kHz).
- Channels: mono (1 channel).
- Encoding: 16-bit PCM WAV.
- `metadata.csv`: one row per audio file with label, split, format, impact onset and generation notes.
- Files are generated with a fixed random seed for reproducibility.

## Folder layout
```
train/
  plastic/
  metal/
validation/
  plastic/
  metal/
test/
  plastic/
  metal/
metadata.csv
class_counts.csv
README.md
```

## Generation method
Each two-second waveform contains a simulated impact after a short lead-in randomized between 35 and 90 ms. It combines decaying sinusoidal resonances, a short click/thump, decaying noise, background noise, and a low-frequency room-like component.
- Metal: stronger, longer-lasting narrow-band ringing.
- Plastic: more damped resonances and a rounded thump.
These are deliberately simplified, hand-designed profiles, not measurements or a validated physical simulation of a metal/plastic object striking the actual impact plate.

## Parameter glossary
- Sample rate 16,000 Hz: 16,000 samples per second; Nyquist frequency is 8,000 Hz. This matches the rate expected by the existing hardware recorder, but matching the sample rate alone does not make the synthetic files representative of the microphone.
- Duration 2 seconds: 32,000 samples per clip. At signed 16-bit mono PCM this is 64,000 bytes, matching the recorder's documented frame size.
- Mono: one channel, appropriate for the single INMP441 microphone.
- 16-bit PCM: stored sample precision; not a guarantee of sensor accuracy.
- Label: `plastic` or `metal`.
- Split: training fits model parameters; validation selects settings; test is held back. These splits only evaluate this synthetic generator.
- Impact onset: randomized simulated event start time.
- Noise parameter: standard deviation used for generic noise before peak normalization. It is not a calibrated microphone noise level or real-world SNR.
- `synthetic=True` and `physical_recording=False` mark that these are simulated, not recorded from hardware.

## Limitations
1. Synthetic-to-real gap: no clips were recorded from the actual plate, INMP441, ESP32 I2S chain or room.
2. Hand-designed class profiles make metal ring longer and plastic sound more damped. A classifier can learn these artificial rules rather than real material signatures.
3. No object identity or real object diversity: no actual cans, tools, bottles, containers, masses, shapes, thicknesses, or contents.
4. Noise is generic and not calibrated to a classroom or workshop.
5. Peak normalization scales clips to similar peak range, so absolute impact loudness is not preserved as a reliable feature.
6. Real I2S artifacts, microphone frequency response, mounting resonance, clipping, servo/electrical noise and plate vibration are not modeled accurately.
7. High accuracy on the synthetic test set is not evidence of real-world sorter accuracy.
8. Although 16 kHz and two seconds match the documented recorder format, the earlier microphone diagnostics reported sparse/near-silent samples. Do not train the production hardware model on these synthetic files or on bad real captures.

## Recommended use
Use this dataset to verify WAV loading, waveform/spectrogram plotting, feature extraction and model code. Once the microphone produces clear, repeatable real audio, collect multiple physical objects per class and split by object/session. Train and evaluate the production model on verified real recordings. Keep synthetic and real evaluation results separate.
