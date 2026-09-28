from pathlib import Path
import shutil

SRC = Path(r"D:\SatQueryAI\data\processed\grounding_images")
DST = Path(r"D:\SatQueryAI\kaggle\reference_grounding_end_to_end")

print("=" * 70)
print("PREPARING KAGGLE END-TO-END GROUNDING DATASET")
print("=" * 70)

if not SRC.exists():
    raise FileNotFoundError(SRC)

DST.mkdir(parents=True, exist_ok=True)

# Copy manifests
for name in [
    "reference_train.parquet",
    "reference_val.parquet",
]:
    src = SRC / name
    dst = DST / name

    if not src.exists():
        raise FileNotFoundError(src)

    shutil.copy2(src, dst)
    print(f"Manifest ready: {name}")

# Copy the already-created S1 ZIP only if you want to retain it locally.
# It is NOT used by the direct-upload dataset.
old_zip = DST / "BigEarthNet-S1.zip"

if old_zip.exists():
    print(f"\nRemoving obsolete local archive: {old_zip}")
    old_zip.unlink()

print("\nThe dataset will be uploaded from:")
print(SRC)

print("\nWe will upload these directories directly:")
print("  BigEarthNet-S1/")
print("  BigEarthNet-S2/")
print("  reference_train.parquet")
print("  reference_val.parquet")

print("\nNo ZIP compression will be performed.")
print("=" * 70)