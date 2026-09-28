import json
from pathlib import Path

VOCAB_PATH = Path(
    r"D:\SatQueryAI\data\processed\c3_semantic_targets\clc_vocabulary.json"
)

print("=" * 70)
print("INSPECT C3 CLC VOCABULARY")
print("=" * 70)
print()

print("Path:")
print(VOCAB_PATH)
print()

if not VOCAB_PATH.exists():
    raise FileNotFoundError(
        f"Vocabulary file not found:\n{VOCAB_PATH}"
    )

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

print("JSON Python type:")
print(type(data).__name__)
print()

if isinstance(data, dict):

    print("TOP-LEVEL KEYS:")
    for key in data.keys():
        print(f"  {key}")

    print()
    print("TOP-LEVEL VALUES / TYPES:")
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            print(
                f"  {key}: "
                f"{type(value).__name__}, "
                f"length={len(value)}"
            )
        else:
            print(
                f"  {key}: "
                f"{type(value).__name__}"
            )

elif isinstance(data, list):

    print(
        f"List length: {len(data)}"
    )

    print()
    print("FIRST 10 ITEMS:")

    for item in data[:10]:
        print(
            f"  {repr(item)} "
            f"(type={type(item).__name__})"
        )

else:

    print("VALUE:")
    print(repr(data))


print()
print("RAW JSON:")
print("-" * 70)
print(
    json.dumps(
        data,
        indent=2,
    )
)

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)