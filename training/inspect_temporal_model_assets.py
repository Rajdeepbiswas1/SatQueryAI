from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path(r"D:\SatQueryAI")

RAW_CDVQA = ROOT / "data" / "raw" / "CDVQA"
TEMPORAL_FEATURES = ROOT / "data" / "processed" / "temporal_features"
TEMPORAL_PREPROCESSED = ROOT / "data" / "processed" / "temporal_preprocessed"
TEMPORAL_MANIFEST = ROOT / "data" / "processed" / "temporal_manifest.csv"

SOURCE_DIRS = [
    ROOT / "src" / "models",
    ROOT / "training",
    ROOT / "preprocessing",
]

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "temporal_model_inspection.txt"


SEARCH_TERMS = [
    "cdvqa",
    "temporal",
    "change",
    "difference",
    "lstm",
    "gru",
    "rnn",
    "transformer",
    "question",
    "answer",
    "classifier",
    "softmax",
    "torch.load",
    "checkpoint",
]


def write(lines: list[str], message: str = "") -> None:
    print(message)
    lines.append(message)


def section(lines: list[str], title: str) -> None:
    write(lines, "")
    write(lines, "=" * 80)
    write(lines, title)
    write(lines, "=" * 80)


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def describe_value(value: Any, indent: str = "    ") -> list[str]:
    result = []

    if torch.is_tensor(value):
        result.append(
            f"{indent}Tensor shape={tuple(value.shape)} "
            f"dtype={value.dtype} "
            f"device={value.device}"
        )
        if value.numel() > 0 and torch.is_floating_point(value):
            finite = torch.isfinite(value).all().item()
            result.append(f"{indent}finite={finite}")
            result.append(
                f"{indent}min={value.min().item():.6f} "
                f"max={value.max().item():.6f} "
                f"mean={value.mean().item():.6f}"
            )
        return result

    if isinstance(value, dict):
        result.append(f"{indent}dict keys={list(value.keys())}")
        for key, item in value.items():
            result.append(f"{indent}key={key!r}")
            result.extend(describe_value(item, indent + "    "))
        return result

    if isinstance(value, list):
        result.append(f"{indent}list length={len(value)}")
        if value:
            result.append(f"{indent}first item type={type(value[0]).__name__}")
            if isinstance(value[0], (str, int, float)):
                result.append(f"{indent}first item={value[0]!r}")
        return result

    if isinstance(value, tuple):
        result.append(f"{indent}tuple length={len(value)}")
        return result

    result.append(
        f"{indent}{type(value).__name__}: {repr(value)[:300]}"
    )
    return result


def inspect_feature_shards(lines: list[str]) -> None:
    section(lines, "1. TEMPORAL DOFA FEATURE CACHE")

    if not TEMPORAL_FEATURES.exists():
        write(lines, f"NOT FOUND: {TEMPORAL_FEATURES}")
        return

    shards = sorted(TEMPORAL_FEATURES.glob("shard_*.pt"))

    write(lines, f"Directory: {TEMPORAL_FEATURES}")
    write(lines, f"Shard count: {len(shards)}")

    for shard in shards:
        write(
            lines,
            f"{shard.name}: {shard.stat().st_size:,} bytes"
        )

    if not shards:
        write(lines, "No feature shards found.")
        return

    first = shards[0]

    write(lines, "")
    write(lines, f"Loading first shard only: {first.name}")

    try:
        obj = safe_torch_load(first)

        write(lines, f"Loaded type: {type(obj).__name__}")

        if isinstance(obj, dict):
            write(lines, f"Keys: {list(obj.keys())}")

            for key, value in obj.items():
                if torch.is_tensor(value):
                    write(
                        lines,
                        f"{key}: shape={tuple(value.shape)}, "
                        f"dtype={value.dtype}, "
                        f"numel={value.numel():,}"
                    )
                elif isinstance(value, list):
                    write(
                        lines,
                        f"{key}: list length={len(value)}"
                    )
                    if value:
                        write(
                            lines,
                            f"{key}[0]: {repr(value[0])[:200]}"
                        )
                else:
                    write(
                        lines,
                        f"{key}: type={type(value).__name__}, "
                        f"value={repr(value)[:200]}"
                    )
        else:
            write(lines, "Unexpected shard structure:")
            for line in describe_value(obj):
                write(lines, line)

    except Exception as exc:
        write(
            lines,
            f"ERROR loading feature shard: "
            f"{type(exc).__name__}: {exc}"
        )


def inspect_temporal_sample(lines: list[str]) -> None:
    section(lines, "2. TEMPORAL PREPROCESSED SAMPLE")

    if not TEMPORAL_PREPROCESSED.exists():
        write(lines, f"NOT FOUND: {TEMPORAL_PREPROCESSED}")
        return

    files = sorted(TEMPORAL_PREPROCESSED.glob("*.pt"))

    write(lines, f"Directory: {TEMPORAL_PREPROCESSED}")
    write(lines, f"File count: {len(files)}")

    if not files:
        return

    sample = files[0]
    write(lines, f"Loading sample: {sample.name}")

    try:
        obj = safe_torch_load(sample)

        write(lines, f"Loaded type: {type(obj).__name__}")

        if isinstance(obj, dict):
            write(lines, f"Keys: {list(obj.keys())}")

            for key, value in obj.items():
                if torch.is_tensor(value):
                    write(
                        lines,
                        f"{key}: shape={tuple(value.shape)}, "
                        f"dtype={value.dtype}"
                    )
                else:
                    write(
                        lines,
                        f"{key}: type={type(value).__name__}, "
                        f"value={repr(value)[:300]}"
                    )
        else:
            for line in describe_value(obj):
                write(lines, line)

    except Exception as exc:
        write(
            lines,
            f"ERROR loading temporal sample: "
            f"{type(exc).__name__}: {exc}"
        )


def inspect_manifest(lines: list[str]) -> None:
    section(lines, "3. TEMPORAL MANIFEST")

    if not TEMPORAL_MANIFEST.exists():
        write(lines, f"NOT FOUND: {TEMPORAL_MANIFEST}")
        return

    write(lines, f"File: {TEMPORAL_MANIFEST}")
    write(
        lines,
        f"Size: {TEMPORAL_MANIFEST.stat().st_size:,} bytes"
    )

    try:
        with TEMPORAL_MANIFEST.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.DictReader(f)

            fieldnames = reader.fieldnames or []
            write(lines, f"Columns: {fieldnames}")

            row_count = 0
            counters: dict[str, Counter] = {}

            interesting_columns = [
                "type",
                "category",
                "answer",
                "output",
                "input",
                "patch_id",
                "id",
                "image_id",
                "split",
            ]

            for column in interesting_columns:
                if column in fieldnames:
                    counters[column] = Counter()

            first_rows = []

            for row in reader:
                row_count += 1

                if len(first_rows) < 5:
                    first_rows.append(row)

                for column, counter in counters.items():
                    value = row.get(column)
                    if value is not None:
                        counter[value] += 1

            write(lines, f"Total manifest rows: {row_count}")

            for idx, row in enumerate(first_rows, start=1):
                write(lines, "")
                write(lines, f"Sample row {idx}:")
                for key, value in row.items():
                    write(
                        lines,
                        f"    {key}: {repr(value)[:500]}"
                    )

            for column, counter in counters.items():
                write(lines, "")
                write(lines, f"Top values for '{column}':")

                for value, count in counter.most_common(25):
                    write(
                        lines,
                        f"    {count:6d}  {value!r}"
                    )

    except Exception as exc:
        write(
            lines,
            f"ERROR reading manifest: "
            f"{type(exc).__name__}: {exc}"
        )


def inspect_cdvqa_json(lines: list[str]) -> None:
    section(lines, "4. OFFICIAL CDVQA JSON SCHEMA")

    if not RAW_CDVQA.exists():
        write(lines, f"NOT FOUND: {RAW_CDVQA}")
        return

    json_files = sorted(RAW_CDVQA.glob("*.json"))

    write(lines, f"Directory: {RAW_CDVQA}")
    write(lines, f"JSON file count: {len(json_files)}")

    for path in json_files:
        write(
            lines,
            f"{path.name}: {path.stat().st_size:,} bytes"
        )

    for split in ["Train", "Val", "Test", "Test2"]:
        section_name = f"{split} JSON SET"

        question_file = RAW_CDVQA / f"{split}_questions.json"
        answer_file = RAW_CDVQA / f"{split}_answers.json"
        image_file = RAW_CDVQA / f"{split}_images.json"

        if not (
            question_file.exists()
            and answer_file.exists()
            and image_file.exists()
        ):
            continue

        write(lines, "")
        write(lines, section_name)

        try:
            with question_file.open(
                "r",
                encoding="utf-8"
            ) as f:
                questions = json.load(f)

            with answer_file.open(
                "r",
                encoding="utf-8"
            ) as f:
                answers = json.load(f)

            with image_file.open(
                "r",
                encoding="utf-8"
            ) as f:
                images = json.load(f)

            write(
                lines,
                f"questions type={type(questions).__name__}, "
                f"length={len(questions) if hasattr(questions, '__len__') else 'N/A'}"
            )

            write(
                lines,
                f"answers type={type(answers).__name__}, "
                f"length={len(answers) if hasattr(answers, '__len__') else 'N/A'}"
            )

            write(
                lines,
                f"images type={type(images).__name__}, "
                f"length={len(images) if hasattr(images, '__len__') else 'N/A'}"
            )

            if questions:
                write(lines, "")
                write(lines, "First question item:")
                first_q = questions[0]
                write(
                    lines,
                    json.dumps(
                        first_q,
                        indent=2,
                        ensure_ascii=False
                    )[:2500]
                )

            if answers:
                write(lines, "")
                write(lines, "First answer item:")
                first_a = answers[0]
                write(
                    lines,
                    json.dumps(
                        first_a,
                        indent=2,
                        ensure_ascii=False
                    )[:2500]
                )

            if images:
                write(lines, "")
                write(lines, "First image item:")
                first_i = images[0]
                write(
                    lines,
                    json.dumps(
                        first_i,
                        indent=2,
                        ensure_ascii=False
                    )[:2500]
                )

        except Exception as exc:
            write(
                lines,
                f"ERROR reading {split}: "
                f"{type(exc).__name__}: {exc}"
            )


def inspect_temporal_source(lines: list[str]) -> None:
    section(lines, "5. EXISTING TEMPORAL SOURCE")

    target = ROOT / "src" / "data" / "temporal_dataset.py"

    write(lines, f"Target: {target}")

    if not target.exists():
        write(lines, "temporal_dataset.py NOT FOUND")
    else:
        try:
            source = target.read_text(
                encoding="utf-8"
            )

            write(
                lines,
                f"Source size: {len(source):,} characters"
            )

            tree = ast.parse(source)

            classes = []
            functions = []

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)

            write(lines, f"Top-level classes: {classes}")
            write(lines, f"Top-level functions: {functions}")

            write(lines, "")
            write(lines, "Imported modules:")

            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        write(lines, f"    import {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(
                        alias.name
                        for alias in node.names
                    )
                    write(
                        lines,
                        f"    from {module} import {names}"
                    )

        except Exception as exc:
            write(
                lines,
                f"ERROR parsing temporal_dataset.py: "
                f"{type(exc).__name__}: {exc}"
            )


def scan_relevant_source(lines: list[str]) -> None:
    section(lines, "6. SOURCE CODE SEARCH FOR TEMPORAL MODEL LOGIC")

    matched_files = 0

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue

        for path in sorted(source_dir.rglob("*.py")):
            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
            except Exception:
                continue

            lower = text.lower()

            matched_terms = [
                term
                for term in SEARCH_TERMS
                if term.lower() in lower
            ]

            if not matched_terms:
                continue

            matched_files += 1

            write(lines, "")
            write(lines, f"FILE: {path}")
            write(lines, f"MATCHES: {matched_terms}")

            source_lines = text.splitlines()

            shown = 0

            for line_no, line in enumerate(
                source_lines,
                start=1
            ):
                low = line.lower()

                if any(
                    term.lower() in low
                    for term in SEARCH_TERMS
                ):
                    write(
                        lines,
                        f"    {line_no:5d}: "
                        f"{line[:350]}"
                    )

                    shown += 1

                    if shown >= 20:
                        write(
                            lines,
                            "    ... further matches omitted for this file ..."
                        )
                        break

    write(lines, "")
    write(lines, f"Relevant Python files matched: {matched_files}")


def inspect_checkpoints(lines: list[str]) -> None:
    section(lines, "7. CHECKPOINT SEARCH")

    if not CHECKPOINT_DIR.exists():
        write(lines, f"NOT FOUND: {CHECKPOINT_DIR}")
        return

    all_checkpoints = sorted(
        CHECKPOINT_DIR.rglob("*.pt")
    )

    write(
        lines,
        f"Total .pt checkpoint files: "
        f"{len(all_checkpoints)}"
    )

    temporal_matches = []

    terms = [
        "temporal",
        "cdvqa",
        "change",
    ]

    for path in all_checkpoints:
        lower = str(path).lower()

        if any(term in lower for term in terms):
            temporal_matches.append(path)

    write(
        lines,
        f"Temporal/change-related checkpoint candidates: "
        f"{len(temporal_matches)}"
    )

    for path in temporal_matches:
        write(
            lines,
            f"    {path} "
            f"({path.stat().st_size:,} bytes)"
        )

    write(lines, "")
    write(lines, "All checkpoint directories:")

    directories = sorted(
        {
            path.parent
            for path in all_checkpoints
        }
    )

    for directory in directories:
        write(lines, f"    {directory}")


def inspect_git(lines: list[str]) -> None:
    section(lines, "8. LOCAL CDVQA GIT STATE")

    if not RAW_CDVQA.exists():
        return

    git_dir = RAW_CDVQA / ".git"

    if not git_dir.exists():
        write(lines, "CDVQA is not a Git working tree.")
        return

    commands = [
        ["git", "-C", str(RAW_CDVQA), "remote", "-v"],
        ["git", "-C", str(RAW_CDVQA), "branch", "--show-current"],
        ["git", "-C", str(RAW_CDVQA), "log", "-5", "--oneline"],
        ["git", "-C", str(RAW_CDVQA), "status", "--short"],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            write(lines, "")
            write(lines, "$ " + " ".join(command))

            stdout = result.stdout.strip()

            if stdout:
                for line in stdout.splitlines():
                    write(lines, "    " + line)

            stderr = result.stderr.strip()

            if stderr:
                for line in stderr.splitlines():
                    write(lines, "    STDERR: " + line)

        except Exception as exc:
            write(
                lines,
                f"Git command failed: "
                f"{type(exc).__name__}: {exc}"
            )


def inspect_dataset_file_names(lines: list[str]) -> None:
    section(lines, "9. TEMPORAL FEATURE ID CONSISTENCY")

    if not TEMPORAL_PREPROCESSED.exists():
        write(lines, "Temporal preprocessed directory missing.")
        return

    sample_names = sorted(
        path.stem
        for path in TEMPORAL_PREPROCESSED.glob("*.pt")
    )[:30]

    write(
        lines,
        f"First 30 temporal pair IDs: {sample_names}"
    )

    feature_files = sorted(
        TEMPORAL_FEATURES.glob("*.pt")
    )

    if feature_files:
        try:
            obj = safe_torch_load(feature_files[0])

            if isinstance(obj, dict):
                for key in ["patch_ids", "pair_ids", "ids", "indices"]:
                    if key in obj:
                        value = obj[key]

                        if isinstance(value, list):
                            write(
                                lines,
                                f"Feature shard '{key}' "
                                f"length={len(value)}"
                            )
                            write(
                                lines,
                                f"First IDs: {value[:10]}"
                            )
                        elif torch.is_tensor(value):
                            write(
                                lines,
                                f"Feature shard '{key}': "
                                f"shape={tuple(value.shape)}, "
                                f"dtype={value.dtype}"
                            )

        except Exception as exc:
            write(
                lines,
                f"Could not inspect feature IDs: "
                f"{type(exc).__name__}: {exc}"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    section(lines, "SATQUERYAI — TEMPORAL MODEL ASSET INSPECTION")

    write(lines, f"Project root: {ROOT}")
    write(lines, "Purpose: inspect existing temporal assets and implementations.")
    write(lines, "No training will be performed.")
    write(lines, "No checkpoints will be modified.")
    write(lines, "Only the first feature shard and first preprocessed sample are loaded.")

    inspect_feature_shards(lines)
    inspect_temporal_sample(lines)
    inspect_manifest(lines)
    inspect_cdvqa_json(lines)
    inspect_temporal_source(lines)
    scan_relevant_source(lines)
    inspect_checkpoints(lines)
    inspect_git(lines)
    inspect_dataset_file_names(lines)

    section(lines, "10. INSPECTION COMPLETE")

    write(lines, "No model was trained.")
    write(lines, "No checkpoint was modified.")
    write(lines, f"Full report saved to: {OUTPUT_FILE}")

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("")
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print(f"REPORT: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()