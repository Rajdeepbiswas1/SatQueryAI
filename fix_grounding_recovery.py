import ast
import json
from pathlib import Path

PROJECT = Path(r"D:\SatQueryAI")
NOTEBOOK = PROJECT / "kaggle_downloads" / "notebooke1eb76f0a0.ipynb"
OUT_DIR = PROJECT / "src" / "models" / "grounding_recovered"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_notebook():
    with NOTEBOOK.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_code_cells(notebook):
    result = []

    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        if not source.strip():
            continue

        result.append((idx, source))

    return result


def safe_parse(source):
    """
    Parse only cells that are valid standalone Python.

    Kaggle/Jupyter cells may contain:
      !pip ...
      %time ...
      %%capture ...
    Those are notebook commands, not Python syntax.
    """
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def find_class_source(source, class_name):
    tree = safe_parse(source)

    if tree is None:
        return None

    lines = source.splitlines(keepends=True)

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            start = node.lineno - 1
            end = node.end_lineno
            return "".join(lines[start:end])

    return None


def find_function_source(source, function_name):
    tree = safe_parse(source)

    if tree is None:
        return None

    lines = source.splitlines(keepends=True)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                start = node.lineno - 1
                end = node.end_lineno
                return "".join(lines[start:end])

    return None


notebook = load_notebook()
cells = get_code_cells(notebook)

print(f"Notebook: {NOTEBOOK}")
print(f"Code cells found: {len(cells)}")
print()

point_source = None
c3_source = None
helper_source = {}

point_cell = None
c3_cell = None
helper_cells = {}

for idx, source in cells:

    if point_source is None:
        found = find_class_source(
            source,
            "PointGroundingModel"
        )

        if found:
            point_source = found
            point_cell = idx
            print(
                f"PointGroundingModel found in cell {idx}"
            )

    if c3_source is None:
        found = find_class_source(
            source,
            "C3QueryConditionedGrounding"
        )

        if found:
            c3_source = found
            c3_cell = idx
            print(
                f"C3QueryConditionedGrounding found in cell {idx}"
            )

    for name in (
        "heatmap_moments",
        "heatmap_to_box",
    ):
        if name not in helper_source:
            found = find_function_source(
                source,
                name
            )

            if found:
                helper_source[name] = found
                helper_cells[name] = idx
                print(
                    f"{name} found in cell {idx}"
                )


if point_source is None:
    raise RuntimeError(
        "PointGroundingModel was not found in any "
        "Python-parsable notebook cell."
    )

if c3_source is None:
    raise RuntimeError(
        "C3QueryConditionedGrounding was not found in any "
        "Python-parsable notebook cell."
    )

for name in (
    "heatmap_moments",
    "heatmap_to_box",
):
    if name not in helper_source:
        raise RuntimeError(
            f"Required helper not found: {name}"
        )


# ------------------------------------------------------------
# Write Point model
# ------------------------------------------------------------

point_output = '''"""
Recovered PointGroundingModel.

Recovered from the saved SatQueryAI Kaggle notebook.
Only the model class itself is included.
Notebook-level data paths, training code, and Kaggle commands
are intentionally excluded.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

''' + point_source + "\n"

point_path = OUT_DIR / "point_grounding_recovered.py"
point_path.write_text(
    point_output,
    encoding="utf-8"
)


# ------------------------------------------------------------
# Write C3 model and helpers
# ------------------------------------------------------------

c3_output = '''"""
Recovered C3QueryConditionedGrounding.

Recovered from the saved SatQueryAI Kaggle notebook.
Only the mathematical helper functions and model class are
included.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''

c3_output += helper_source["heatmap_moments"]
c3_output += "\n\n"
c3_output += helper_source["heatmap_to_box"]
c3_output += "\n\n"
c3_output += c3_source
c3_output += "\n"

c3_path = OUT_DIR / "c3_reference_grounding_recovered.py"
c3_path.write_text(
    c3_output,
    encoding="utf-8"
)


# ------------------------------------------------------------
# Preserve exact original class/function snippets
# ------------------------------------------------------------

(OUT_DIR / "point_source_original_cell.py").write_text(
    point_source,
    encoding="utf-8"
)

(OUT_DIR / "c3_source_original_cell.py").write_text(
    c3_source,
    encoding="utf-8"
)

for name, source in helper_source.items():
    (
        OUT_DIR / f"{name}_original_cell.py"
    ).write_text(
        source,
        encoding="utf-8"
    )


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

manifest = {
    "notebook": str(NOTEBOOK),
    "point_cell": point_cell,
    "c3_cell": c3_cell,
    "helper_cells": helper_cells,
    "helpers_found": list(helper_source.keys()),
    "outputs": [
        str(point_path),
        str(c3_path),
    ],
}

(OUT_DIR / "recovery_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)


print()
print("=" * 72)
print("GROUNDING SOURCE RECOVERY COMPLETE")
print("=" * 72)
print(f"Point cell: {point_cell}")
print(f"C3 cell:    {c3_cell}")

for name in (
    "heatmap_moments",
    "heatmap_to_box",
):
    print(
        f"{name}: "
        f"FOUND (cell {helper_cells[name]})"
    )

print()
print(f"Point source:")
print(f"  {point_path}")

print()
print(f"C3 source:")
print(f"  {c3_path}")

print("=" * 72)