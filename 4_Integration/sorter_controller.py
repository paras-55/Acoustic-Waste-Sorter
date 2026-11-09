"""
sorter_controller.py - drives the whole sorter. Serial Monitor must be CLOSED.

Modes (run from the project folder):
  python 4_Integration\\sorter_controller.py test
        Serial integration test: centre, plastic, metal, centre - prints each ACK.
  python 4_Integration\\sorter_controller.py angle
        Type angles (e.g. 40, 140) to find the best tilt positions for your tray.
  python 4_Integration\\sorter_controller.py live [--ask-truth]
        REAL acoustic ML: drop object -> mic -> model -> servo. Logs to 4_Test_Results.
  python 4_Integration\\sorter_controller.py manual
        FALLBACK, clearly labelled: operator types p/m, no audio, no ML.
  python 4_Integration\\sorter_controller.py replay <wav files or folder>
        FALLBACK, clearly labelled: model classifies PRE-RECORDED clips, servo sorts.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_Integration"))
sys.path.insert(0, str(ROOT / "3_ML_Model"))

from esp_link import EspLink, EspError, PORT, SAMPLE_RATE  # noqa: E402

RESULTS = ROOT / "4_Test_Results"


def banner(text):
    print("=" * 64)
    print(text)
    print("=" * 64)


def mode_test(esp):
    banner("SERIAL INTEGRATION TEST")
    for lab in ["center", "plastic", "metal", "center"]:
        t = time.time()
        ack = esp.sort(lab)
        print(f"  {lab:8s} -> {ack}  ({time.time() - t:.2f}s)")
    print("PASS: laptop -> ESP32 -> servo link works.")


def mode_angle(esp):
    banner("ANGLE CALIBRATION - type an angle 0-180, or q to quit")
    print("Find the plastic-side and metal-side angles, then edit ANGLE_PLASTIC / ANGLE_METAL")
    print("in WasteSorter.ino and re-upload.")
    while True:
        s = input("angle> ").strip().lower()
        if s in ("q", "quit", ""):
            esp.sort("center")
            return
        if s.isdigit():
            print("  ", esp.angle(int(s)))


def mode_manual(esp):
    banner("MANUAL MODE - operator selects the class. NO audio, NO machine learning.")
    while True:
        s = input("p = plastic, m = metal, c = centre, q = quit > ").strip().lower()
        if s == "q":
            return
        lab = {"p": "plastic", "m": "metal", "c": "center"}.get(s)
        if lab:
            print("  ", esp.sort(lab))


def mode_replay(esp, paths):
    from predict_audio import Classifier
    clf = Classifier()
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.wav")) if p.is_dir() else [p]
    banner(f"REPLAY MODE - PRE-RECORDED audio, not live microphone.\nModel {clf.version} - trained on: {clf.trained_on}")
    for f in files:
        input(f"Enter to classify {f.name} ...")
        r = clf.predict_file(f)
        act = r["label"] if r["label"] in ("plastic", "metal") else "center"
        vs = f"vote share {r['vote_share']:.2f}" if r["vote_share"] is not None else "; ".join(r["reasons"])
        print(f"  {f.name}: {r['label'].upper()} ({vs}) -> {esp.sort(act)}")


def mode_live(esp, ask_truth):
    from predict_audio import Classifier
    from audio_features import int16_to_float, save_wav
    clf = Classifier()
    d = esp.diagnostic()
    if not d["ok"]:
        print("Microphone diagnostic FAILED - live mode unavailable. Use 'manual' or 'replay' and say so in the demo.")
        return
    clips = RESULTS / "live_clips"
    clips.mkdir(parents=True, exist_ok=True)
    log = RESULTS / "live_log.csv"
    header = ["time", "clip", "predicted", "vote_share", "true_label", "snr_db", "peak", "model_version", "true_object"]
    if log.exists():                                   # add the true_object column to older logs
        with open(log, newline="") as fr:
            rows = list(csv.reader(fr))
        if rows and "true_object" not in rows[0]:
            with open(log, "w", newline="") as fw:
                cw = csv.writer(fw)
                cw.writerow(header)
                for row in rows[1:]:
                    cw.writerow(row + [""] * (len(header) - len(row)))
    new = not log.exists()
    f = open(log, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(header)

    banner(f"LIVE ACOUSTIC CLASSIFICATION - microphone + ML model {clf.version}\nModel trained on: {clf.trained_on}")
    n = correct = unsure = 0
    if ask_truth:
        print("After each drop, type the TRUE material: m or p - or with the object number, e.g. m3 or p2.")
        print("Answers with an object number become training data for: python 3_ML_Model\\train_model.py --include-live")
    try:
        while True:
            if input("\nEnter = arm, q = quit > ").strip().lower() == "q":
                break
            esp.sort("center")
            print("  Armed - drop the object on the tray now.")
            try:
                samples, status = esp.record(triggered=True)
            except EspError as e:
                print("  Transfer error:", e)
                continue
            if samples is None:
                print("  No impact detected (timeout).")
                continue
            x = int16_to_float(samples)
            r = clf.predict_array(x, SAMPLE_RATE)
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
            clip = clips / f"live_{stamp}.wav"
            save_wav(clip, samples)
            if r["label"] in ("plastic", "metal"):
                print(f"  PREDICTION: {r['label'].upper()}  (model score {r['vote_share']:.2f}, not calibrated)")
                print("  ", esp.sort(r["label"]))
            elif r["label"] == "UNSURE":
                print(f"  UNSURE (best guess {r['best_guess']}, score {r['vote_share']:.2f}) - tray stays centred.")
            else:
                print(f"  REJECTED clip: {'; '.join(r['reasons'])} - drop again.")
            truth, obj = "", ""
            if ask_truth and r["label"] != "REJECTED":
                ans = input("  True material? m/p, or with object number e.g. m3 / Enter=skip > ").strip().lower()
                if ans[:1] in ("m", "p"):
                    truth = "metal" if ans[0] == "m" else "plastic"
                    obj = ans[1:] if ans[1:].isdigit() else ""
                    n += 1
                    if r["label"] == "UNSURE":
                        unsure += 1
                    else:
                        correct += truth == r["label"]
                    print(f"  Live so far: {correct}/{n} correct ({correct / n:.0%}), {unsure} unsure, "
                          f"{n - correct - unsure} wrong")
            st = r.get("stats", {})
            w.writerow([stamp, clip.name, r["label"], None if r.get("vote_share") is None else round(r["vote_share"], 3), truth,
                        round(st.get("snr_db", 0), 1), round(st.get("peak", 0), 4), clf.version, obj])
            f.flush()
    finally:
        f.close()
        esp.sort("center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test", "angle", "live", "manual", "replay"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--ask-truth", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    try:
        with EspLink(args.port, verbose=False) as esp:
            if args.mode == "test":
                mode_test(esp)
            elif args.mode == "angle":
                mode_angle(esp)
            elif args.mode == "manual":
                mode_manual(esp)
            elif args.mode == "replay":
                if not args.files:
                    print("Give WAV files or a folder, e.g. replay 2_Audio_Dataset\\Metal")
                    return 1
                mode_replay(esp, args.files)
            else:
                mode_live(esp, args.ask_truth)
    except EspError as e:
        print("ERROR:", e)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
