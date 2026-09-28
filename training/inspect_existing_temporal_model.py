from pathlib import Path
import ast

ROOT = Path(r"D:\SatQueryAI")

TARGETS = [
    ROOT / "src" / "models" / "feature_fusion.py",
    ROOT / "src" / "data" / "temporal_dataset.py",
    ROOT / "preprocessing" / "test_dofa_temporal_one.py",
    ROOT / "preprocessing" / "preprocess_temporal_dataset.py",
    ROOT / "preprocessing" / "verify_temporal_dataset.py",
]

SEARCH_TERMS = [
    "TemporalFusion",
    "Temporal",
    "change",
    "difference",
    "question",
    "answer",
    "classifier",
    "loss",
    "forward",
    "dataset",
]


def inspect_file(path: Path) -> None:
    print()
    print("=" * 90)
    print(f"FILE: {path}")
    print("=" * 90)

    if not path.exists():
        print("STATUS: NOT FOUND")
        return

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    print(f"SIZE: {len(text):,} characters")
    print()

    try:
        tree = ast.parse(text)

        classes = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        ]

        functions = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        ]

        print("TOP-LEVEL CLASSES:")
        for name in classes:
            print(f"  {name}")

        print()

        print("TOP-LEVEL FUNCTIONS:")
        for name in functions:
            print(f"  {name}")

    except Exception as exc:
        print(
            "AST PARSE ERROR:",
            type(exc).__name__,
            str(exc),
        )

    print()
    print("RELEVANT SOURCE LINES:")

    source_lines = text.splitlines()

    for i, line in enumerate(source_lines, start=1):
        low = line.lower()

        if any(term.lower() in low for term in SEARCH_TERMS):
            print(f"{i:5d}: {line}")

    print()
    print("FULL SOURCE:")
    print("-" * 90)
    print(text)
    print("-" * 90)


def main() -> None:
    print("=" * 90)
    print("SATQUERYAI — EXISTING TEMPORAL MODEL INSPECTION")
    print("=" * 90)
    print("This script does not train anything.")
    print("This script does not modify checkpoints.")
    print("Purpose: inspect the temporal model components already present.")
    print()

    for target in TARGETS:
        inspect_file(target)

    print()
    print("=" * 90)
    print("INSPECTION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()