from pathlib import Path
import zipfile
import shutil

ROOT = Path(r"D:\SatQueryAI\data\processed\grounding_images")
OUT = Path(r"D:\SatQueryAI\kaggle\reference_grounding_end_to_end")

S1_SRC = ROOT / "BigEarthNet-S1"
S2_SRC = ROOT / "BigEarthNet-S2"

S1_ZIP = OUT / "BigEarthNet-S1.zip"
S2_ZIP = OUT / "BigEarthNet-S2.zip"

print("=" * 70)
print("SATQUERYAI — PACKAGE END-TO-END GROUNDING DATASET")
print("=" * 70)

OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Copy manifests
# ------------------------------------------------------------------

for name in [
    "reference_train.parquet",
    "reference_val.parquet",
]:
    src = ROOT / name
    dst = OUT / name

    if not src.exists():
        raise FileNotFoundError(src)

    shutil.copy2(src, dst)
    print(f"Copied: {name}")

# ------------------------------------------------------------------
# Zip a directory while preserving its directory structure
# ------------------------------------------------------------------

def create_zip(source_dir: Path, zip_path: Path):
    if zip_path.exists():
        print(f"\nRemoving existing archive: {zip_path.name}")
        zip_path.unlink()

    files = [
        p for p in source_dir.rglob("*")
        if p.is_file()
    ]

    print(f"\nCreating {zip_path.name}")
    print(f"Files: {len(files):,}")

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:

        for i, src in enumerate(files, 1):
            # Archive path starts inside the source directory.
            arcname = Path(source_dir.name) / src.relative_to(source_dir)

            zf.write(src, arcname)

            if i % 5000 == 0 or i == len(files):
                print(
                    f"  Added {i:,}/{len(files):,}"
                )

    size_gib = zip_path.stat().st_size / (1024 ** 3)

    print(
        f"Created: {zip_path.name} "
        f"({size_gib:.3f} GiB)"
    )

    return len(files)


# ------------------------------------------------------------------
# Package S1
# ------------------------------------------------------------------

s1_count = create_zip(S1_SRC, S1_ZIP)

# ------------------------------------------------------------------
# Package S2
# ------------------------------------------------------------------

s2_count = create_zip(S2_SRC, S2_ZIP)

# ------------------------------------------------------------------
# Final check
# ------------------------------------------------------------------

expected_s1 = 7602 * 2
expected_s2 = 7602 * 12

print("\n" + "=" * 70)
print("PACKAGE SUMMARY")
print("=" * 70)

print(f"S1 files: {s1_count:,} / expected {expected_s1:,}")
print(f"S2 files: {s2_count:,} / expected {expected_s2:,}")

assert s1_count == expected_s1
assert s2_count == expected_s2

total_bytes = sum(
    p.stat().st_size
    for p in OUT.iterdir()
    if p.is_file()
)

print(
    f"\nTotal package size: "
    f"{total_bytes / (1024 ** 3):.3f} GiB"
)

print("\nOutput directory:")
print(OUT)

print("\nContents:")
for p in sorted(OUT.iterdir()):
    size = p.stat().st_size / (1024 ** 3)
    print(f"  {p.name:30s} {size:.3f} GiB")

print("\nAll packaging checks passed.")
print("=" * 70)