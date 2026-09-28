from pathlib import Path


PROJECT_ROOT = Path(r"D:\SatQueryAI")

S1_LIST = PROJECT_ROOT / "data" / "raw" / "s1_extract_list.txt"
S2_LIST = PROJECT_ROOT / "data" / "raw" / "s2_extract_list.txt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "images_raw"


def check_list(list_file: Path):
    members = [
        line.strip()
        for line in list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    missing = []

    for member in members:
        path = OUTPUT_DIR / Path(member)

        if not path.is_file():
            missing.append(member)

    return len(members), missing


print("=== Verifying extracted imagery ===\n")

s1_total, s1_missing = check_list(S1_LIST)
s2_total, s2_missing = check_list(S2_LIST)

print(f"S1 requested: {s1_total:,}")
print(f"S1 missing:   {len(s1_missing):,}")

print(f"\nS2 requested: {s2_total:,}")
print(f"S2 missing:   {len(s2_missing):,}")

print("\n=== Result ===")

if not s1_missing and not s2_missing:
    print("SUCCESS: Every requested S1 and S2 file exists.")
else:
    print("WARNING: Some requested files are missing.")

    if s1_missing:
        print("\nMissing S1 files:")
        for x in s1_missing[:20]:
            print(x)

        if len(s1_missing) > 20:
            print(f"... and {len(s1_missing) - 20:,} more")

    if s2_missing:
        print("\nMissing S2 files:")
        for x in s2_missing[:20]:
            print(x)

        if len(s2_missing) > 20:
            print(f"... and {len(s2_missing) - 20:,} more")