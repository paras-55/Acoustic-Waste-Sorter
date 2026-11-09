
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent / "2_Audio_Dataset"

for material in ["Plastic", "Metal"]:
    print(f"\n--- {material} ---")

    for path in sorted((ROOT / material).glob("*.wav")):
        audio, sr = sf.read(path)

        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))

        print(
            f"{path.name:18s} "
            f"peak={peak:.6f}  "
            f"RMS={rms:.6f}"
        )
