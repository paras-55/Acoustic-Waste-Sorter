
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent / "2_Audio_Dataset"

for material in ("Metal", "Plastic"):
    path = ROOT / material / f"{material.lower()}_065.wav"
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    peak = np.max(np.abs(audio))
    threshold = max(peak * 0.10, 0.001)
    active = np.flatnonzero(np.abs(audio) > threshold)

    print(f"\n{path.name}")
    print(f"Sample rate: {sr}")
    print(f"Duration: {len(audio) / sr:.2f}s")
    print(f"Peak: {peak:.6f}")
    print(f"RMS: {np.sqrt(np.mean(audio ** 2)):.6f}")

    if len(active):
        print(f"First active sample: {active[0]} ({active[0]/sr:.3f}s)")
        print(f"Last active sample: {active[-1]} ({active[-1]/sr:.3f}s)")
        print(f"Active duration: {(active[-1]-active[0]+1)/sr:.3f}s")
    else:
        print("No samples above threshold.")
