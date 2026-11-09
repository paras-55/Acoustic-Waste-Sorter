"""
record_audio.py - quality-gated dataset recorder (replaces the old version).

Usage (from the project folder, Serial Monitor CLOSED):
    python 2_Audio_Dataset\\record_audio.py metal 1
    python 2_Audio_Dataset\\record_audio.py plastic 3 --count 10

  label    : plastic | metal
  object   : number of the PHYSICAL object (write it on the object with a marker).
             Use a different number for every different object, per material.
  --count  : how many ACCEPTED clips to collect (default 10). Rejected ones are retried.
  --gain   : optional gain shift 12..18 sent to the ESP32 (default: firmware's 15;
             smaller = louder). Use the same value for the whole dataset.

Every clip is checked by the shared quality gate (3_ML_Model/audio_features.py).
Only clips that pass are saved to Plastic/ or Metal/ as  <label>_objNN_<seq>.wav.
Rejected clips go to _rejected/ (never used for training) and every attempt is
logged in recordings_log.csv.
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "4_Integration"))
sys.path.insert(0, str(ROOT / "3_ML_Model"))

from esp_link import EspLink, EspError, PORT, SAMPLE_RATE  # noqa: E402
from audio_features import quality, int16_to_float, save_wav  # noqa: E402

FOLDERS = {"plastic": HERE / "Plastic", "metal": HERE / "Metal"}
REJECT_DIR = HERE / "_rejected"
LOG = HERE / "recordings_log.csv"
LOG_FIELDS = ["time", "file", "label", "object_id", "accepted", "reasons",
              "peak", "event_rms", "noise_rms", "snr_db", "zero_frac", "clip_frac", "gain_shift"]


def next_seq(folder, label):
    """Next sequence number after ANY existing <label>_..._NNN.wav, so nothing is overwritten."""
    best = 0
    for p in folder.glob(f"{label}_*.wav"):
        m = re.search(r"_(\d+)\.wav$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def log_row(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", choices=["plastic", "metal"])
    ap.add_argument("object_id", type=int)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--gain", type=int, default=None)
    ap.add_argument("--port", default=PORT)
    args = ap.parse_args()

    folder = FOLDERS[args.label]
    folder.mkdir(exist_ok=True)
    REJECT_DIR.mkdir(exist_ok=True)

    print(f"Connecting to ESP32 on {args.port} ...")
    try:
        esp = EspLink(args.port, verbose=False)
    except EspError as e:
        print("ERROR:", e)
        return 1

    with esp:
        d = esp.diagnostic()
        if not d["ok"]:
            print("Microphone diagnostic FAILED - not recording. Run 4_Integration\\mic_check.py.")
            for l in d["lines"]:
                print("  ", l)
            return 2
        gain = "default"
        if args.gain is not None:
            gain = esp.set_gain_shift(args.gain).split()[-1]
        print(f"Mic OK (slot {d['slot']}), gain shift {gain}.")
        print(f"Recording {args.count} accepted clips of {args.label.upper()} object #{args.object_id}.")
        print("For each clip: press Enter, then drop the object on the plate within 15 s. 'q' + Enter quits.\n")

        accepted = attempts = 0
        while accepted < args.count:
            if input(f"[{accepted}/{args.count}] Enter = arm trigger: ").strip().lower() == "q":
                break
            attempts += 1
            try:
                samples, status = esp.record(triggered=True)
            except EspError as e:
                print("   REJECTED: transfer error -", e)
                continue
            if samples is None:
                print("   REJECTED: no impact detected (timeout)")
                continue

            x = int16_to_float(samples)
            ok, st, reasons = quality(x, SAMPLE_RATE)
            seq = next_seq(folder, args.label)
            name = f"{args.label}_obj{args.object_id:02d}_{seq:03d}.wav"
            target = (folder if ok else REJECT_DIR) / name
            while target.exists():                       # belt and braces: never overwrite
                seq += 1
                name = f"{args.label}_obj{args.object_id:02d}_{seq:03d}.wav"
                target = (folder if ok else REJECT_DIR) / name
            save_wav(target, samples)

            print(f"   peak={st['peak']:.3f} event_rms={st['event_rms']:.4f} SNR={st['snr_db']:.1f}dB "
                  f"zeros={100*st['zero_frac']:.1f}% clipped={100*st['clip_frac']:.2f}%")
            if ok:
                accepted += 1
                print(f"   ACCEPTED -> {target.relative_to(ROOT)}")
            else:
                print(f"   REJECTED ({'; '.join(reasons)}) -> kept in _rejected/ for inspection")
            log_row({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "file": str(target.relative_to(ROOT)),
                     "label": args.label, "object_id": args.object_id, "accepted": ok,
                     "reasons": "; ".join(reasons), "peak": round(st.get("peak", 0), 5),
                     "event_rms": round(st.get("event_rms", 0), 5), "noise_rms": round(st.get("noise_rms", 0), 6),
                     "snr_db": round(st.get("snr_db", 0), 2), "zero_frac": round(st.get("zero_frac", 0), 4),
                     "clip_frac": round(st.get("clip_frac", 0), 5), "gain_shift": gain})

        print(f"\nDone: {accepted} accepted out of {attempts} attempts for {args.label} object #{args.object_id}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
