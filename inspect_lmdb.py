from pathlib import Path

root = Path(r"D:\SatQueryAI\data\raw\BigEarthNet_LMDB")

print("=" * 70)
print("BIGEARTHNET LMDB INSPECTION")
print("=" * 70)

if not root.exists():
    print("LMDB directory does not exist.")
    raise SystemExit

print("\nPath:")
print(root)

print("\nContents:")

total_files = 0
total_size = 0

for p in sorted(root.rglob("*")):
    if p.is_file():
        total_files += 1
        size = p.stat().st_size
        total_size += size

        print(
            f"  {p.relative_to(root)}"
            f"    {size / (1024**2):.2f} MB"
        )

print("\n" + "-" * 70)
print(f"Total files: {total_files:,}")
print(f"Total size:  {total_size / (1024**3):.3f} GB")
print("=" * 70)