"""
mic_check.py - ONE run that tells you whether the microphone pipeline works.

  1. Runs the firmware diagnostic (both I2S slots) and prints the verdict.
  2. Arms the trigger: drop/tap an object on the plate when told.
  3. Runs the quality gate on that clip, saves it to
     4_Test_Results/mic_check.wav  (+ mic_check.png waveform if matplotlib exists)

Run (from the project folder, Serial Monitor CLOSED):
    python 4_Integration\\mic_check.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_Integration"))
sys.path.insert(0, str(ROOT / "3_ML_Model"))

from esp_link import EspLink, EspError, PORT, SAMPLE_RATE  # noqa: E402
from audio_features import quality, int16_to_float, save_wav  # noqa: E402

OUT_DIR = ROOT / "4_Test_Results"


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else PORT
    print(f"Connecting to ESP32 on {port} ...")
    try:
        esp = EspLink(port)
    except EspError as e:
        print("ERROR:", e)
        return 1

    with esp:
        print("\n=== Step 1: I2S slot diagnostic (keep quiet for 1 s) ===")
        d = esp.diagnostic()
        if not d["ok"]:
            print("\nVERDICT: MIC_FAULT - no I2S slot carries a live microphone signal.")
            print("Send me the DIAG lines above. Do NOT record a dataset yet.")
            return 2
        print(f"\nVERDICT: microphone stream is live on slot {d['slot']}.")

        print("\n=== Step 2: impact test ===")
        print("Trigger armed for 15 s - drop a METAL object on the plate now.")
        try:
            samples, status = esp.record(triggered=True)
        except EspError as e:
            print("ERROR:", e)
            return 1
        if samples is None:
            print("No impact detected within 15 s (trigger timeout). Drop it closer / harder and re-run.")
            return 3

    x = int16_to_float(samples)
    ok, st, reasons = quality(x, SAMPLE_RATE)
    OUT_DIR.mkdir(exist_ok=True)
    wav = OUT_DIR / "mic_check.wav"
    save_wav(wav, samples)          # always saved here (test folder, not the dataset)

    print(f"\nClip: {st['duration_s']:.2f}s  peak={st['peak']:.3f}  event_rms={st['event_rms']:.4f}  "
          f"noise_rms={st['noise_rms']:.5f}  SNR={st['snr_db']:.1f} dB  "
          f"zeros={100*st['zero_frac']:.1f}%  clipped={100*st['clip_frac']:.2f}%")
    print(f"Saved: {wav}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        t = np.arange(len(x)) / SAMPLE_RATE
        plt.figure(figsize=(10, 3))
        plt.plot(t, x, lw=0.6)
        plt.xlabel("time (s)"); plt.ylabel("amplitude"); plt.title("mic_check.wav")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "mic_check.png", dpi=120)
        print(f"Saved: {OUT_DIR / 'mic_check.png'}")
    except Exception:
        pass

    if ok:
        print("\nRESULT: PASS - clear impact captured. Listen to mic_check.wav, then start recording the dataset.")
        return 0
    print("\nRESULT: FAIL")
    for r in reasons:
        print("  -", r)
    return 4


if __name__ == "__main__":
    sys.exit(main())
