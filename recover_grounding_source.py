import ast
import json
from pathlib import Path

PROJECT = Path(r"D:\SatQueryAI")
NOTEBOOK = PROJECT / "kaggle_downloads" / "notebooke1eb76f0a0.ipynb"

OUT_DIR = PROJECT / "src" / "models" / "grounding_recovered"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load saved Kaggle notebook
# ------------------------------------------------------------
if not NOTEBOOK.exists():
    raise FileNotFoundError(f"Saved Kaggle notebook not found:\n{NOTEBOOK}")

with NOTEBOOK.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

cells = notebook.get("cells", [])

code_cells = []
for i, cell in enumerate(cells):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if source.strip():
            code_cells.append((i, source))

print(f"Notebook: {NOTEBOOK}")
print(f"Code cells found: {len(code_cells)}")


# ------------------------------------------------------------
# Search helpers
# ------------------------------------------------------------
def find_cells(*terms):
    hits = []
    for cell_index, source in code_cells:
        if all(term in source for term in terms):
            hits.append((cell_index, source))
    return hits


def find_cells_any(*terms):
    hits = []
    for cell_index, source in code_cells:
        if any(term in source for term in terms):
            hits.append((cell_index, source))
    return hits


def write_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"WRITTEN: {path}")


# ------------------------------------------------------------
# Locate exact model cells
# ------------------------------------------------------------
point_hits = find_cells("class PointGroundingModel")
c3_hits = find_cells("class C3QueryConditionedGrounding")

if not point_hits:
    raise RuntimeError("Could not find class PointGroundingModel in saved notebook.")

if not c3_hits:
    raise RuntimeError(
        "Could not find class C3QueryConditionedGrounding in saved notebook."
    )

point_cell_index, point_source = point_hits[0]
c3_cell_index, c3_source = c3_hits[0]

print(f"PointGroundingModel cell: {point_cell_index}")
print(f"C3QueryConditionedGrounding cell: {c3_cell_index}")


# ------------------------------------------------------------
# Find exact helper functions required by C3
# ------------------------------------------------------------
helper_names = [
    "heatmap_moments",
    "heatmap_to_box",
]

helper_sources = {}

for helper_name in helper_names:
    hits = find_cells(f"def {helper_name}")
    if hits:
        helper_sources[helper_name] = hits[0][1]
        print(f"Helper found: {helper_name} (cell {hits[0][0]})")
    else:
        print(f"WARNING: helper not found: {helper_name}")


# ------------------------------------------------------------
# Find shared/import cells likely required by the models
# ------------------------------------------------------------
shared_candidates = []

keywords = [
    "import torch",
    "import torch.nn",
    "from torch",
    "class ",
    "def ",
    "nn.",
    "F.",
]

for cell_index, source in code_cells:
    # We only recover cells that are plausibly model-definition cells.
    if (
        "PointGroundingModel" in source
        or "C3QueryConditionedGrounding" in source
        or "heatmap_moments" in source
        or "heatmap_to_box" in source
        or "Cross" in source
        or "Grounding" in source
    ):
        shared_candidates.append((cell_index, source))


# ------------------------------------------------------------
# Extract AST-level top-level definitions from model cells.
# This avoids executing the notebook.
# ------------------------------------------------------------
def extract_top_level_defs(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    pieces = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
            ),
        ):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            pieces.append("".join(lines[start:end]))

    return "\n\n".join(pieces)


# ------------------------------------------------------------
# Recover Point model
# ------------------------------------------------------------
point_defs = extract_top_level_defs(point_source)

point_file = '''"""
Recovered Point Grounding model from the saved SatQueryAI Kaggle notebook.

This file is source recovery only.
No training code is executed here.
"""

import math
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

'''

point_file += point_defs + "\n"

write_file(
    OUT_DIR / "point_grounding_recovered.py",
    point_file,
)


# ------------------------------------------------------------
# Recover C3 reference model
# ------------------------------------------------------------
c3_defs = extract_top_level_defs(c3_source)

c3_file = '''"""
Recovered C3 Reference Grounding model from the saved SatQueryAI Kaggle notebook.

This file is source recovery only.
No training code is executed here.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

'''

for name in helper_names:
    if name in helper_sources:
        c3_file += extract_top_level_defs(helper_sources[name]) + "\n\n"

c3_file += c3_defs + "\n"

write_file(
    OUT_DIR / "c3_reference_grounding_recovered.py",
    c3_file,
)


# ------------------------------------------------------------
# Save the original full source cells for forensic comparison.
# ------------------------------------------------------------
write_file(
    OUT_DIR / "point_source_original_cell.py",
    point_source,
)

write_file(
    OUT_DIR / "c3_source_original_cell.py",
    c3_source,
)

for name, source in helper_sources.items():
    write_file(
        OUT_DIR / f"{name}_original_cell.py",
        source,
    )


# ------------------------------------------------------------
# Save a manifest
# ------------------------------------------------------------
manifest = {
    "notebook": str(NOTEBOOK),
    "point_cell": point_cell_index,
    "c3_cell": c3_cell_index,
    "helpers": {
        name: name in helper_sources
        for name in helper_names
    },
    "outputs": [
        str(OUT_DIR / "point_grounding_recovered.py"),
        str(OUT_DIR / "c3_reference_grounding_recovered.py"),
        str(OUT_DIR / "point_source_original_cell.py"),
        str(OUT_DIR / "c3_source_original_cell.py"),
    ],
}

write_file(
    OUT_DIR / "recovery_manifest.json",
    json.dumps(manifest, indent=2),
)


print()
print("=" * 70)
print("GROUNDING SOURCE RECOVERY COMPLETE")
print("=" * 70)
print(f"Point cell:       {point_cell_index}")
print(f"C3 cell:          {c3_cell_index}")
print(f"heatmap_moments:  {'FOUND' if 'heatmap_moments' in helper_sources else 'MISSING'}")
print(f"heatmap_to_box:   {'FOUND' if 'heatmap_to_box' in helper_sources else 'MISSING'}")
print(f"Output directory: {OUT_DIR}")
print()