
from pathlib import Path
import soundfile as sf

ROOT = Path(__file__).parent
EXPECTED_RATE = 16000

total = 0
errors = []

for material in ("Plastic", "Metal"):
    folder = ROOT / material
    files = sorted(folder.glob("*.wav"))

    print(f"\n{material}: {len(files)} recordings")
    total += len(files)

    object_counts = {}

    for file in files:
        try:
            audio, sample_rate = sf.read(file)
            object_id = "_".join(file.stem.split("_")[:-1])
            object_counts[object_id] = object_counts.get(object_id, 0) + 1

            if sample_rate != EXPECTED_RATE:
                errors.append(
                    f"{file.name}: sample rate is {sample_rate}"
                )

            if len(audio) == 0:
                errors.append(f"{file.name}: empty audio")

        except Exception as error:
            errors.append(f"{file.name}: {error}")

    print("Objects found:", len(object_counts))
    print("Samples per object:", sorted(object_counts.values()))

print(f"\nTotal recordings: {total}")

if errors:
    print("\nIssues found:")
    for error in errors:
        print("-", error)
else:
    print("No basic audio-file issues found.")
