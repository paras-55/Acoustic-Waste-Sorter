# Acoustic Waste Sorter

## Sound-Based Waste Classification Using Acoustic Sensing and Machine Learning

An intelligent waste sorting system that identifies **metal and plastic objects based on the sound they produce when they impact a surface**.

The project combines **acoustic sensing, signal processing, machine learning, and embedded hardware** to create a working prototype for automated waste classification.

---

## Overview

Traditional waste sorting systems often rely on cameras, manual sorting, or expensive industrial sensors. This project explores an alternative approach: **using sound as a source of information**.

When an object is dropped onto a standardized impact plate, different materials produce different acoustic signatures. A microphone captures the impact sound, which is processed and analyzed by a machine learning model.

The system currently classifies waste into two categories:

- **Metal**
- **Plastic**

The complete prototype integrates an ESP32-based audio capture system with a trained machine learning model and a software dashboard for monitoring the classification process.

---

## How It Works

The overall workflow of the system is:

```text
Object Dropped
      ↓
Impact on Standardized Plate
      ↓
Microphone Captures Sound
      ↓
ESP32 Transmits Audio Data
      ↓
Audio Processing & Feature Extraction
      ↓
Machine Learning Classification
      ↓
Metal or Plastic Prediction
```

### Step-by-Step Process

1. A waste object is dropped onto a standardized impact surface.
2. The impact produces a unique sound pattern.
3. A microphone captures the acoustic signal.
4. The ESP32 digitizes and transmits the audio data to the computer.
5. The recorded audio is checked and processed.
6. Acoustic features are extracted from the sound.
7. The trained machine learning model analyzes these features.
8. The system predicts whether the object is **metal or plastic**.
9. The result can be monitored through the integrated software dashboard.

---

## Project Features

- Real-world acoustic recordings for material classification
- Classification of **metal and plastic**
- ESP32-based audio acquisition
- Standardized impact-based testing
- Audio feature extraction and analysis
- Machine learning-based classification
- Quality checking for recorded audio
- Prediction on new audio samples
- Integrated monitoring dashboard
- Test recordings and training results included

---

# Hardware Used

The prototype uses the following main hardware components:

| Component | Purpose |
|---|---|
| ESP32 | Controls audio acquisition and communication |
| Microphone Module | Captures the sound produced during impact |
| Impact Plate | Provides a standardized surface for object impacts |
| MG996R Servo Motor | Used as part of the physical sorting mechanism |
| Breadboard | Hardware prototyping and circuit connections |
| Jumper Wires | Electrical connections between components |
| Power Supply | Powers the hardware components |

### Audio Configuration

The system was developed using the following audio settings:

- **Sample Rate:** 16 kHz
- **Recording Duration:** Approximately 2 seconds
- **Audio Processing:** Python-based
- **Communication:** Serial communication between ESP32 and computer

---

# Machine Learning Pipeline

The machine learning workflow consists of several stages.

## 1. Data Collection

Real impact sounds were recorded for both material classes.

The dataset was collected using multiple objects for each material to make the model learn the characteristics of the **material**, rather than memorizing a single object.

The recordings are organized into:

- Metal samples
- Plastic samples
- Training data
- Validation data
- Test recordings

## 2. Audio Quality Checking

Audio recordings are inspected before being used for training or prediction.

This helps identify:

- Very quiet recordings
- Recordings without a clear impact
- Noisy or poor-quality samples

Low-quality recordings can be rejected to improve the reliability of the machine learning pipeline.

## 3. Feature Extraction

The audio signal is converted into meaningful numerical features that can be understood by the machine learning model.

The project includes audio analysis and feature extraction through:

- Time-domain analysis
- Frequency-domain analysis
- Spectral features
- Acoustic feature measurements

The extracted features represent characteristics of the impact sound and help distinguish between metal and plastic.

## 4. Model Training

The extracted features are used to train a classification model.

The trained model is saved for later use so that new recordings can be classified without retraining the entire system.

The trained model files include:

```text
waste_classifier.joblib
waste_classifier.meta.json
```

## 5. Prediction

When a new object is tested:

1. Its impact sound is recorded.
2. Features are extracted.
3. The trained model analyzes the features.
4. The system returns a prediction: **Metal or Plastic**.

---

# Project Structure

```text
Acoustic-Waste-Sorter/
│
├── 1_ESP32_Code/
│   ├── AudioRecorder/
│   ├── WasteSorter/
│   └── WasteSorter.ino
│
├── 2_Audio_Dataset/
│   ├── Metal/
│   ├── Plastic/
│   └── synthetic/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── 3_ML_Model/
│   ├── audio_features.py
│   ├── check_audio_levels.py
│   ├── inspect_clips.py
│   ├── plot_clips.py
│   ├── predict_audio.py
│   ├── train_classifier.py
│   ├── train_model.py
│   ├── waste_classifier.joblib
│   └── waste_classifier.meta.json
│
├── 4_Integration/
│   ├── dashboard.html
│   ├── dashboard.py
│   ├── esp_link.py
│   ├── mic_check.py
│   └── sorter_controller.py
│
├── 5_Test_Results/
│   ├── live_clips/
│   ├── live_log.csv
│   ├── mic_check.png
│   ├── mic_check.wav
│   ├── sorter_controller.py
│   └── training_report.txt
│
├── README.md
└── RUN_ORDER.txt
```

---

# Software Requirements

The project uses:

- Python
- Arduino IDE
- ESP32 Board Support Package
- Required Python libraries for:
  - Audio processing
  - Machine learning
  - Data analysis
  - Visualization

Install the required Python dependencies before running the machine learning and integration scripts.

---

# Running the Project

The recommended workflow is:

### 1. Connect the Hardware

Connect the ESP32 and microphone according to the hardware configuration.

Ensure that:

- The ESP32 is detected by the computer.
- The correct serial port is selected.
- The microphone is functioning correctly.

### 2. Upload the ESP32 Code

Open the appropriate Arduino sketch from:

```text
1_ESP32_Code/
```

Upload the code to the ESP32 using the Arduino IDE.

### 3. Verify the Microphone

Use the microphone checking tools to verify that the system is receiving an audio signal.

### 4. Train the Model

If retraining is required, run the training scripts inside:

```text
3_ML_Model/
```

The trained classifier will be saved as a `.joblib` model file.

### 5. Run Prediction and Integration

Use the scripts in:

```text
4_Integration/
```

to connect the ESP32 audio stream, machine learning model, and monitoring dashboard.

For the recommended execution sequence, refer to:

```text
RUN_ORDER.txt
```

---

# Testing

The project includes test recordings and results inside:

```text
5_Test_Results/
```

These files contain recorded audio clips and logs generated during system testing.

Testing focuses on evaluating how well the trained model performs on recordings and objects that were not necessarily used directly during training.

---

# Key Concept

The core idea behind this project is that different materials produce different **acoustic signatures** when they strike a surface.

For example:

- A **metal object** may produce a sharper and more resonant impact sound.
- A **plastic object** may produce a comparatively damped or lower-resonance sound.

Instead of relying on visual appearance, the system analyzes these acoustic differences to identify the material.

This approach demonstrates the potential of **passive acoustic sensing** as an alternative method for material classification.

---

# Future Improvements

The current prototype focuses on **metal and plastic**. Possible future improvements include:

- Adding more material categories such as glass, cardboard, and paper
- Increasing the size and diversity of the audio dataset
- Improving classification accuracy with additional feature engineering
- Exploring deep learning models for audio classification
- Deploying the ML model directly on an embedded device
- Improving the physical sorting mechanism
- Developing a more advanced real-time monitoring interface

---

# Project Status

**Status: Fully Operational Prototype**

The project currently demonstrates the complete pipeline from:

**Physical Impact → Sound Capture → Audio Processing → Feature Extraction → Machine Learning Classification → Result Monitoring**

The prototype serves as a proof of concept for using **acoustic sensing and machine learning for intelligent waste classification**.

---

## Contributors

Developed as an academic innovation project.

---

## License

This project is intended for educational and research purposes.