
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "2_Audio_Dataset"

for material in ("Metal", "Plastic"):
    path = ROOT / material / f"{material.lower()}_065.wav"
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    time_axis = np.arange(len(audio)) / sr

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio, linewidth=0.6)
    plt.title(f"{material} impact — {path.name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.xlim(0, len(audio) / sr)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path.parent / f"{material.lower()}_065_waveform.png")
    plt.show()
