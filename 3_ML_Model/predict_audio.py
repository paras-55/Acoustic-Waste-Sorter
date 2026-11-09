"""
predict_audio.py - classify WAV files (or arrays) with the validated model.

Run:
    python 3_ML_Model\\predict_audio.py path\\to\\clip.wav [more.wav ...]

Import:
    from predict_audio import Classifier
    clf = Classifier(); result = clf.predict_array(samples_float, 16000)

The 'model score' is the SVM's probability estimate for the class.
It is NOT a calibrated probability; it is shown only as a margin indicator.
"""
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn notice about SVC(probability=True)
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audio_features import FEATURE_VERSION, SAMPLE_RATE, load_wav, quality, extract_features  # noqa: E402

MODEL_PATH = HERE / "waste_classifier.joblib"
MIN_VOTE_SHARE = 0.60          # below this the result is reported as UNSURE


class Classifier:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found. It is created by train_model.py only after validation passes.")
        b = joblib.load(model_path)
        if b.get("feature_version") != FEATURE_VERSION:
            raise RuntimeError("Model was trained with a different feature version - re-run train_model.py")
        self.bundle = b
        self.pipe = b["pipeline"]
        self.classes = list(self.pipe.classes_)
        self.version = b.get("model_version", "?")
        self.trained_on = b.get("trained_on", "real recordings")

    def predict_array(self, x, sr=SAMPLE_RATE):
        """Returns dict: label ('plastic'|'metal'|'UNSURE'|'REJECTED'), vote_share, reasons, stats."""
        ok, st, reasons = quality(np.asarray(x, dtype=np.float32), sr)
        if not ok:
            return {"label": "REJECTED", "vote_share": None, "reasons": reasons, "stats": st}
        f = extract_features(x, sr).reshape(1, -1)
        votes = self.pipe.predict_proba(f)[0]
        k = int(np.argmax(votes))
        label = self.classes[k] if votes[k] >= MIN_VOTE_SHARE else "UNSURE"
        return {"label": label, "best_guess": self.classes[k], "vote_share": float(votes[k]),
                "votes": dict(zip(self.classes, map(float, votes))), "reasons": [], "stats": st}

    def predict_file(self, path):
        x, sr = load_wav(path)
        return self.predict_array(x, sr)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    clf = Classifier()
    print(f"Model version {clf.version} - trained on: {clf.trained_on}")
    for p in sys.argv[1:]:
        r = clf.predict_file(p)
        if r["label"] == "REJECTED":
            print(f"{p}: REJECTED - {'; '.join(r['reasons'])}")
        else:
            print(f"{p}: {r['label'].upper()}  (model score {r['vote_share']:.2f}, not calibrated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
