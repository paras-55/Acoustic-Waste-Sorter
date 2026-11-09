"""
train_model.py - quality-gated, object-grouped training of the plastic/metal classifier.

Run (from the project folder):
    python 3_ML_Model\\train_model.py
    python 3_ML_Model\\train_model.py --include-legacy   # also try old plastic_NNN.wav files
    python 3_ML_Model\\train_model.py --quarantine       # MOVE rejected files to 2_Audio_Dataset\\_quarantine

  Synthetic dataset (folder with train/ validation/ test/ subfolders):
    python 3_ML_Model\\train_model.py --synthetic-only --synthetic 2_Audio_Dataset\\synthetic
        Model trained ONLY on synthetic audio. Labelled as such everywhere; its
        accuracy is measured on synthetic test clips only (not real evidence).
    python 3_ML_Model\\train_model.py --synthetic 2_Audio_Dataset\\synthetic
        HYBRID: real recordings + synthetic clips for training, but validation uses
        ONLY real objects. Needs fewer real clips (>= 12 per class, >= 3 objects).
        Also reports how a synthetic-only model scores on your real clips.

What it does
  1. Loads WAVs from 2_Audio_Dataset/Plastic and /Metal.
     New files  <label>_objNN_<seq>.wav  -> object group from the filename (reliable).
     Old files  <label>_NNN.wav          -> skipped by default (recorded with the broken
                mic stream); with --include-legacy grouped as (NNN-1)//10+1 (assumed).
  2. Rejects bad clips with the shared quality gate and reports every reason.
  3. Requires >= 20 accepted clips and >= 3 distinct objects per class.
  4. Cross-validates with StratifiedGroupKFold: test objects are NEVER seen in training.
  5. Requires out-of-fold accuracy AND balanced accuracy >= 70 %.
  6. Only then trains on everything and atomically replaces waste_classifier.joblib
     (previous model kept as waste_classifier.prev.joblib). If anything fails,
     the existing model is left untouched.
"""
import argparse
import json
import os
import re
import shutil
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn notice about SVC(probability=True)
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC
import csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from audio_features import (GATE, SAMPLE_RATE, FEATURE_NAMES, FEATURE_VERSION,  # noqa: E402
                            load_wav, quality, extract_features)

DATA = ROOT / "2_Audio_Dataset"
CLASSES = ["plastic", "metal"]
MODEL_PATH = HERE / "waste_classifier.joblib"
REPORT_DIR = ROOT / "4_Test_Results"

MIN_CLIPS_PER_CLASS = 20
MIN_OBJECTS_PER_CLASS = 3
MIN_ACCURACY = 0.70
MIN_CLIPS_PER_CLASS_HYBRID = 12

NEW_RE = re.compile(r"^(plastic|metal)_obj(\d+)_(\d+)\.wav$", re.I)
OLD_RE = re.compile(r"^(plastic|metal)_(\d+)\.wav$", re.I)


def collect(include_legacy):
    items, skipped = [], []
    for label in CLASSES:
        folder = DATA / label.capitalize()
        for p in sorted(folder.glob("*.wav")):
            m = NEW_RE.match(p.name)
            if m and m.group(1).lower() == label:
                items.append((p, label, f"{label}-obj{int(m.group(2))}", False))
                continue
            m = OLD_RE.match(p.name)
            if m and m.group(1).lower() == label:
                if include_legacy:
                    obj = (int(m.group(2)) - 1) // 10 + 1
                    items.append((p, label, f"{label}-legacy{obj}", True))
                else:
                    skipped.append((p, "legacy file (old mic stream) - use --include-legacy to test it"))
                continue
            skipped.append((p, "unrecognised filename (expected <label>_objNN_<seq>.wav)"))
    return items, skipped


LIVE_LOG = ROOT / "4_Test_Results" / "live_log.csv"
LIVE_DIR = ROOT / "4_Test_Results" / "live_clips"


def load_live(say):
    """Labelled live drops. With an object number (answered e.g. 'm3') they join that object's
    group and take part in validation; without one they are used for training only."""
    grouped, extra, rej = [], [], 0
    if not LIVE_LOG.exists():
        say("No live_log.csv yet - nothing to include.")
        return grouped, extra
    with open(LIVE_LOG, newline="") as f:
        for row in csv.DictReader(f):
            lab = (row.get("true_label") or "").strip().lower()
            if lab not in CLASSES:
                continue
            p = LIVE_DIR / (row.get("clip") or "")
            if not p.is_file():
                continue
            x, sr = load_wav(p)
            if not quality(x, sr)[0]:
                rej += 1
                continue
            obj = (row.get("true_object") or "").strip()
            feats = extract_features(x, sr)
            if obj.isdigit():
                grouped.append((feats, lab, f"{lab}-obj{int(obj)}"))
            else:
                extra.append((feats, lab))
    say(f"Live drops: {len(grouped)} with object number (used in validation), "
        f"{len(extra)} without (training only), {rej} rejected by the gate")
    return grouped, extra


def load_synthetic(folder, splits, say):
    """Return (X, y, n_rejected) for synthetic clips in folder/<split>/<label>/*.wav."""
    X, y, rej = [], [], 0
    for split in splits:
        for label in CLASSES:
            for p in sorted((folder / split / label).glob("*.wav")):
                x, sr = load_wav(p)
                ok, _, _ = quality(x, sr)
                if not ok:
                    rej += 1
                    continue
                X.append(extract_features(x, sr))
                y.append(label)
    if not X:
        raise SystemExit(f"No synthetic WAVs found under {folder} (expected train/plastic, train/metal, ...)")
    say(f"Synthetic {'+'.join(splits)}: {len(y)} clips accepted, {rej} rejected by the gate  {dict(Counter(y))}")
    return np.vstack(X), np.array(y), rej


def make_model():
    # SVM on standardised, level-independent features: chosen because it scored best on
    # the user's real labelled live drops (Random Forest leaned on loudness and failed live).
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
    ])


def fit_weighted(Xr, yr, Xs, ys):
    """Fit on real + synthetic, weighting so synthetic clips together count as much as the real ones."""
    if Xs is None or len(ys) == 0:
        return make_model().fit(Xr, yr)
    X = np.vstack([Xr, Xs])
    y = np.concatenate([yr, ys])
    w = np.concatenate([np.ones(len(yr)), np.full(len(ys), len(yr) / len(ys))])
    return make_model().fit(X, y, clf__sample_weight=w)


def save_bundle(final, say, report, extra):
    version = time.strftime("%Y%m%d-%H%M%S")
    bundle = {"pipeline": final, "classes": list(final.classes_), "feature_names": FEATURE_NAMES,
              "feature_version": FEATURE_VERSION, "sample_rate": SAMPLE_RATE, "gate": dict(GATE),
              "model_version": version}
    bundle.update(extra)
    tmp = MODEL_PATH.with_suffix(".tmp.joblib")
    joblib.dump(bundle, tmp)
    joblib.load(tmp)                                  # verify it reads back
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, HERE / "waste_classifier.prev.joblib")
    os.replace(tmp, MODEL_PATH)
    meta = {k: v for k, v in bundle.items() if k != "pipeline"}
    (HERE / "waste_classifier.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    say(f"\nMODEL SAVED: {MODEL_PATH.relative_to(ROOT)}  (version {version})")
    say(f"Trained on: {bundle['trained_on']}")
    (REPORT_DIR / "training_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Report: {(REPORT_DIR / 'training_report.txt').relative_to(ROOT)}")
    return 0


def synthetic_only(folder, say, report):
    say("MODE: SYNTHETIC ONLY - no real recordings used. Accuracy below is on synthetic test clips,")
    say("      which says nothing about real-world accuracy. Measure that live with --ask-truth.")
    Xtr, ytr, _ = load_synthetic(folder, ["train", "validation"], say)
    Xte, yte, _ = load_synthetic(folder, ["test"], say)
    m = make_model().fit(Xtr, ytr)
    pred = m.predict(Xte)
    acc = accuracy_score(yte, pred)
    say(f"Synthetic test accuracy: {acc:.3f}")
    if acc < MIN_ACCURACY:
        say("TRAINING STOPPED: synthetic test accuracy too low - nothing overwritten.")
        return 2
    final = make_model().fit(np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]))
    return save_bundle(final, say, report, {
        "trained_on": "SYNTHETIC audio only (not validated on real recordings)",
        "cv": {"synthetic_test_accuracy": acc},
        "n_clips": {"synthetic": int(len(ytr) + len(yte))}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-legacy", action="store_true")
    ap.add_argument("--quarantine", action="store_true")
    ap.add_argument("--synthetic", type=Path, default=None,
                    help="folder of the synthetic dataset (contains train/ validation/ test/)")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--include-live", action="store_true",
                    help="also learn from live-test drops you labelled (4_Test_Results/live_log.csv)")
    args = ap.parse_args()
    if args.synthetic is not None and not args.synthetic.is_absolute():
        args.synthetic = (Path.cwd() / args.synthetic).resolve()
    if args.synthetic_only and args.synthetic is None:
        ap.error("--synthetic-only needs --synthetic <folder>")

    REPORT_DIR.mkdir(exist_ok=True)
    report = []

    def say(s=""):
        print(s)
        report.append(s)

    say(f"Training run {time.strftime('%Y-%m-%d %H:%M:%S')}  gate={GATE}")
    if args.synthetic_only:
        return synthetic_only(args.synthetic, say, report)
    hybrid = args.synthetic is not None
    min_clips = MIN_CLIPS_PER_CLASS_HYBRID if hybrid else MIN_CLIPS_PER_CLASS
    items, skipped = collect(args.include_legacy)
    for p, why in skipped:
        say(f"SKIP   {p.relative_to(ROOT)}: {why}")

    X, y, groups, rejected = [], [], [], []
    for p, label, group, legacy in items:
        try:
            x, sr = load_wav(p)
        except Exception as e:
            rejected.append((p, label, [f"unreadable: {e}"]))
            continue
        ok, st, reasons = quality(x, sr)
        if not ok:
            rejected.append((p, label, reasons))
            continue
        X.append(extract_features(x, sr))
        y.append(label)
        groups.append(group)

    for p, label, reasons in rejected:
        say(f"REJECT {p.relative_to(ROOT)}: {'; '.join(reasons)}")
    live_extra = []
    if args.include_live:
        live_grouped, live_extra = load_live(say)
        for f_, lab_, g_ in live_grouped:
            X.append(f_); y.append(lab_); groups.append(g_)
    if args.quarantine and rejected:
        for p, label, _ in rejected:
            dest = DATA / "_quarantine" / label.capitalize()
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest / p.name))
        say(f"Moved {len(rejected)} rejected files to 2_Audio_Dataset/_quarantine/")

    counts = Counter(y)
    objs = defaultdict(set)
    for lab, g in zip(y, groups):
        objs[lab].add(g)
    say("")
    for lab in CLASSES:
        say(f"{lab:8s}: {counts.get(lab, 0)} accepted clips from {len(objs[lab])} objects {sorted(objs[lab])}")

    def fail(msg):
        say(f"\nTRAINING STOPPED: {msg}")
        say(f"Existing model {'kept' if MODEL_PATH.exists() else 'not present'} - nothing was overwritten.")
        (REPORT_DIR / "training_report.txt").write_text("\n".join(report), encoding="utf-8")
        return 2

    for lab in CLASSES:
        if counts.get(lab, 0) < min_clips:
            return fail(f"{lab} has {counts.get(lab, 0)} accepted clips, need >= {min_clips}")
        if len(objs[lab]) < MIN_OBJECTS_PER_CLASS:
            return fail(f"{lab} has {len(objs[lab])} objects, need >= {MIN_OBJECTS_PER_CLASS}")

    X = np.vstack(X)
    y = np.array(y)
    groups = np.array(groups)

    Xs = ys = None
    synth_baseline = None
    if live_extra:
        Xs = np.vstack([f_ for f_, _ in live_extra])
        ys = np.array([l_ for _, l_ in live_extra])
    if hybrid:
        say("\nMODE: HYBRID - synthetic clips added to TRAINING only; validation uses real objects only.")
        Xsyn, ysyn, _ = load_synthetic(args.synthetic, ["train", "validation", "test"], say)
        pred_s = make_model().fit(Xsyn, ysyn).predict(X)
        Xs = Xsyn if Xs is None else np.vstack([Xs, Xsyn])
        ys = ysyn if ys is None else np.concatenate([ys, ysyn])
        synth_baseline = accuracy_score(y, pred_s)
        say(f"Synthetic-only model tested on your REAL clips: accuracy {synth_baseline:.3f}  "
            f"(this shows how well the synthetic data transfers to your hardware)")

    n_splits = min(5, min(len(objs[l]) for l in CLASSES))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.empty(len(y), dtype=object)
    say(f"\nGrouped cross-validation: {n_splits} folds, test objects never seen in training")
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = fit_weighted(X[tr], y[tr], Xs, ys)
        oof[te] = m.predict(X[te])
        say(f"  fold {k+1}: test objects {sorted(str(g) for g in set(groups[te]))}  acc={accuracy_score(y[te], oof[te]):.2f}")

    acc = accuracy_score(y, oof)
    bacc = balanced_accuracy_score(y, oof)
    cm = confusion_matrix(y, oof, labels=CLASSES)
    say(f"\nOut-of-fold accuracy:          {acc:.3f}")
    say(f"Out-of-fold balanced accuracy: {bacc:.3f}")
    say("Confusion matrix (rows = true, cols = predicted)   plastic  metal")
    for lab, row in zip(CLASSES, cm):
        say(f"  {lab:8s}                                       {row[0]:5d}  {row[1]:5d}")

    if acc < MIN_ACCURACY or bacc < MIN_ACCURACY:
        return fail(f"validation accuracy {acc:.2f} / balanced {bacc:.2f} below {MIN_ACCURACY:.2f}")

    final = fit_weighted(X, y, Xs, ys)
    extra = {
        "trained_on": ("real recordings + synthetic clips (validated on real objects only)" if hybrid
                       else "real recordings" + (" + labelled live drops" if args.include_live else "")),
        "cv": {"accuracy": acc, "balanced_accuracy": bacc, "n_splits": n_splits,
               "confusion_matrix": cm.tolist(), "labels": CLASSES,
               "synthetic_only_on_real": synth_baseline},
        "n_clips": {k: int(v) for k, v in counts.items()},
        "objects": {k: sorted(v) for k, v in objs.items()},
    }
    return save_bundle(final, say, report, extra)


if __name__ == "__main__":
    sys.exit(main())
