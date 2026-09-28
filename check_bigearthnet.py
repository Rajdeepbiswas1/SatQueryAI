from pathlib import Path

root = Path(r"D:\SatQueryAI")

print("=" * 70)
print("SEARCHING FOR BIGEARTHNET SOURCE DATA")
print("=" * 70)

extensions = {".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg", ".pt"}

matches = []

for p in root.rglob("*"):
    if p.is_file():
        name = p.name.lower()
        suffix = p.suffix.lower()

        if (
            suffix in extensions
            or "bigearthnet" in name
            or "ben" in name
            or "s1" in name
            or "s2" in name
        ):
            matches.append(p)

print(f"\nPotential source files: {len(matches):,}\n")

for p in matches[:200]:
    print(p)

print("\n" + "=" * 70)