
from pathlib import Path
import re
import numpy as np
import soundfile as sf
import librosa
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "2_Audio_Dataset"
MODEL_DIR = Path(__file__).resolve().parent
MODEL_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
MATERIALS = {"Plastic": 0, "Metal": 1}


def extract_features(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=SAMPLE_RATE
        )

    audio = audio.astype(np.float32)

    # Reject silent or nearly silent clips.
    if np.max(np.abs(audio)) < 1e-5:
        raise ValueError(f"Nearly silent recording: {path.name}")

    # MFCCs describe the sound's spectral characteristics.
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13,
        n_fft=1024,
        hop_length=256
    )

    # Summarize each coefficient over time.
    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])


features = []
labels = []
groups = []
files = []

for material, label in MATERIALS.items():
    folder = DATASET / material

    for path in sorted(folder.glob("*.wav")):
        match = re.search(r"_(\d+)$", path.stem)
        if not match:
            raise ValueError(f"Unexpected filename: {path.name}")

        sample_number = int(match.group(1))
        if not 1 <= sample_number <= 60:
            raise ValueError(f"Unexpected sample number: {path.name}")

        # Ten consecutive recordings correspond to one object.
        object_number = (sample_number - 1) // 10 + 1

        features.append(extract_features(path))
        labels.append(label)
        groups.append(f"{material}_object_{object_number}")
        files.append(path.name)

X = np.asarray(features)
y = np.asarray(labels)
groups = np.asarray(groups)

print("Total recordings:", len(y))
print("Feature matrix:", X.shape)
print("Objects:", len(set(groups)))

# Each fold holds out two objects from each material.
# With 12 groups total, GroupKFold makes six folds.
cv = GroupKFold(n_splits=6)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, class_weight="balanced")
)

predictions = cross_val_predict(
    model, X, y, groups=groups, cv=cv
)

print("\nGroup-aware cross-validation results")
print("Accuracy:", round(accuracy_score(y, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(
    y, predictions, target_names=["Plastic", "Metal"]
))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y, predictions))

# Fit the final model on all 120 recordings.
model.fit(X, y)

model_path = MODEL_DIR / "waste_classifier.joblib"
joblib.dump(model, model_path)

print("\nFinal model saved to:", model_path)
print("Training complete.")
