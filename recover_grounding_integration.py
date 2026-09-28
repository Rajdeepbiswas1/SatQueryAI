import ast
import json
from pathlib import Path


PROJECT = Path(r"D:\SatQueryAI")
NOTEBOOK = PROJECT / "kaggle_downloads" / "notebooke1eb76f0a0.ipynb"
OUT_DIR = PROJECT / "src" / "models" / "grounding_recovered"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXACT VERIFIED INTEGRATION DEFINITIONS
# ============================================================

TARGET_CLASSES = {
    "UnifiedGroundingModel",
    "UnifiedGroundingInference",
    "SatQueryAIGroundingRouter",
}


# These are the genuine integration helpers identified in
# the recovered final integration cell.
REQUIRED_HELPERS = {
    "_to_batched",
    "_move_to_model_device",
    "_prepare_s1_s2",
    "_prepare_text",
    "_prepare_point",
    "_normalize_heatmap",
    "_sanitize_bbox",
    "_extract_evidence",
    "_find_named_tensor",
    "_heatmap_moment_box_generic",
}


# The recovered notebook source is known to have these
# important definitions in these cells. We verify them rather
# than blindly assuming the cells are correct.
PREFERRED_CLASS_CELLS = {
    "UnifiedGroundingModel": 59,
    "UnifiedGroundingInference": 66,
    "SatQueryAIGroundingRouter": 66,
}

PREFERRED_HELPER_CELL = 66


SPECIALIST_IMPORT_NAMES = {
    "PointGroundingModel",
    "C3QueryConditionedGrounding",
}


# ============================================================
# LOAD NOTEBOOK
# ============================================================

if not NOTEBOOK.exists():
    raise FileNotFoundError(
        f"Notebook not found:\n{NOTEBOOK}"
    )


with NOTEBOOK.open("r", encoding="utf-8") as f:
    notebook = json.load(f)


cells = notebook.get("cells", [])


print("=" * 72)
print("SATQUERYAI — CLEAN UNIFIED GROUNDING SOURCE RECOVERY")
print("=" * 72)
print(f"Notebook: {NOTEBOOK}")
print(f"Notebook cells: {len(cells)}")
print()


# ============================================================
# PARSE ALL CODE CELLS
# ============================================================

parsed_cells = {}

for index, cell in enumerate(cells):

    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))

    if not source.strip():
        continue

    try:
        tree = ast.parse(source)
    except SyntaxError:
        continue

    parsed_cells[index] = {
        "source": source,
        "tree": tree,
        "lines": source.splitlines(keepends=True),
    }


# ============================================================
# HELPERS
# ============================================================

def node_source(source_lines, node):
    """Return the exact original source for an AST node."""
    return "".join(
        source_lines[node.lineno - 1:node.end_lineno]
    )


def top_level_nodes(tree):
    """Return only top-level executable definitions/imports."""
    return list(tree.body)


def find_class_occurrences(class_name):
    occurrences = []

    for cell_index, info in parsed_cells.items():

        for node in info["tree"].body:

            if (
                isinstance(node, ast.ClassDef)
                and node.name == class_name
            ):
                occurrences.append(
                    {
                        "cell": cell_index,
                        "node": node,
                        "source": info["source"],
                        "lines": info["lines"],
                    }
                )

    return occurrences


def find_function_occurrences(function_name):
    occurrences = []

    for cell_index, info in parsed_cells.items():

        for node in info["tree"].body:

            if (
                isinstance(node, ast.FunctionDef)
                and node.name == function_name
            ):
                occurrences.append(
                    {
                        "cell": cell_index,
                        "node": node,
                        "source": info["source"],
                        "lines": info["lines"],
                    }
                )

    return occurrences


def choose_preferred(
    occurrences,
    preferred_cell=None,
    label="definition",
):
    if not occurrences:
        raise RuntimeError(
            f"Could not find {label}."
        )

    if preferred_cell is not None:

        preferred = [
            item
            for item in occurrences
            if item["cell"] == preferred_cell
        ]

        if preferred:
            return preferred[-1]

    # If the preferred cell isn't found, use the latest
    # occurrence because later notebook definitions are the
    # authoritative versions during recovery.
    return occurrences[-1]


def imported_local_names(import_node):
    names = set()

    for alias in import_node.names:

        if alias.name == "*":
            continue

        local_name = (
            alias.asname
            or alias.name.split(".")[0]
        )

        names.add(local_name)

    return names


def collect_used_names(definition_sources):
    combined = "\n\n".join(definition_sources)

    tree = ast.parse(combined)

    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
    }


# ============================================================
# LOCATE EXACT CLASSES
# ============================================================

selected_classes = {}

print("LOCATING INTEGRATION CLASSES")
print("-" * 72)

for class_name in (
    "UnifiedGroundingModel",
    "UnifiedGroundingInference",
    "SatQueryAIGroundingRouter",
):

    occurrences = find_class_occurrences(class_name)

    print(
        f"{class_name}: "
        f"{len(occurrences)} occurrence(s)"
    )

    for occurrence in occurrences:
        print(
            f"    cell {occurrence['cell']}"
        )

    selected = choose_preferred(
        occurrences,
        PREFERRED_CLASS_CELLS.get(class_name),
        label=f"class {class_name}",
    )

    selected_classes[class_name] = selected

    print(
        f"    SELECTED: cell {selected['cell']}"
    )
    print()


# ============================================================
# LOCATE EXACT INTEGRATION HELPERS
# ============================================================

selected_helpers = {}

print("LOCATING INTEGRATION HELPERS")
print("-" * 72)

for helper_name in sorted(REQUIRED_HELPERS):

    occurrences = find_function_occurrences(
        helper_name
    )

    print(
        f"{helper_name}: "
        f"{len(occurrences)} occurrence(s)"
    )

    for occurrence in occurrences:
        print(
            f"    cell {occurrence['cell']}"
        )

    selected = choose_preferred(
        occurrences,
        PREFERRED_HELPER_CELL,
        label=f"helper {helper_name}",
    )

    selected_helpers[helper_name] = selected

    print(
        f"    SELECTED: cell {selected['cell']}"
    )
    print()


# ============================================================
# EXTRACT EXACT SOURCE
# ============================================================

unified_model_source = node_source(
    selected_classes["UnifiedGroundingModel"]["lines"],
    selected_classes["UnifiedGroundingModel"]["node"],
)

inference_source = node_source(
    selected_classes["UnifiedGroundingInference"]["lines"],
    selected_classes["UnifiedGroundingInference"]["node"],
)

router_source = node_source(
    selected_classes["SatQueryAIGroundingRouter"]["lines"],
    selected_classes["SatQueryAIGroundingRouter"]["node"],
)


# Helpers must remain in their original notebook order.
helper_items = sorted(
    selected_helpers.items(),
    key=lambda item: (
        item[1]["cell"],
        item[1]["node"].lineno,
    ),
)

helper_sources = [
    node_source(
        item["lines"],
        item["node"],
    )
    for _, item in helper_items
]


# ============================================================
# COLLECT IMPORTS FROM RELEVANT CELLS ONLY
# ============================================================

relevant_cell_indices = {
    selected_classes["UnifiedGroundingModel"]["cell"],
    selected_classes["UnifiedGroundingInference"]["cell"],
    selected_classes["SatQueryAIGroundingRouter"]["cell"],
}

for _, item in selected_helpers.items():
    relevant_cell_indices.add(item["cell"])


relevant_definition_sources = [
    unified_model_source,
    *helper_sources,
    inference_source,
    router_source,
]

used_names = collect_used_names(
    relevant_definition_sources
)


selected_import_lines = []
future_import_lines = []


for cell_index in sorted(relevant_cell_indices):

    info = parsed_cells[cell_index]

    for node in info["tree"].body:

        if not isinstance(
            node,
            (ast.Import, ast.ImportFrom),
        ):
            continue

        exact_source = node_source(
            info["lines"],
            node,
        ).strip()

        # Preserve future imports explicitly.
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        ):
            if exact_source not in future_import_lines:
                future_import_lines.append(
                    exact_source
                )
            continue

        imported_names = imported_local_names(node)

        # Never pull the specialist classes from the
        # notebook. They must come from our verified
        # local recovered specialist modules.
        imported_names -= SPECIALIST_IMPORT_NAMES

        if not imported_names:
            continue

        if imported_names & used_names:

            if exact_source not in selected_import_lines:
                selected_import_lines.append(
                    exact_source
                )


# ============================================================
# STANDARD IMPORTS
# ============================================================

standard_imports = [
    "import math",
    "import time",
    "import torch",
    "import torch.nn as nn",
    "import torch.nn.functional as F",
]


# Remove exact duplicates from dynamically recovered imports.
filtered_imports = []

standard_import_set = set(standard_imports)

for import_line in selected_import_lines:

    if import_line in standard_import_set:
        continue

    if import_line not in filtered_imports:
        filtered_imports.append(import_line)


# ============================================================
# BUILD CLEAN MODULE
# ============================================================

output_parts = []

output_parts.append(
    '''"""
SatQueryAI Unified Grounding Integration

CLEAN SOURCE RECOVERY

This file contains ONLY:

- required imports
- verified specialist imports
- integration helper functions
- UnifiedGroundingModel
- UnifiedGroundingInference
- SatQueryAIGroundingRouter

Excluded intentionally:

- notebook runtime assignments
- datasets
- DataLoaders
- training code
- checkpoint loading
- evaluation state
- temporary notebook variables
- model instances
- Kaggle runtime state
"""
'''
)


# Future imports must appear before normal imports.
for line in future_import_lines:

    output_parts.append(
        f"\n{line}\n"
    )


output_parts.append("\n")

for line in standard_imports:

    output_parts.append(
        f"{line}\n"
    )


for line in filtered_imports:

    output_parts.append(
        f"\n{line}\n"
    )


# Verified local specialist implementations.
output_parts.append(
    """
from point_grounding_recovered import PointGroundingModel
from c3_reference_grounding_recovered import (
    C3QueryConditionedGrounding,
)
"""
)


# ============================================================
# DEFINITIONS
# ============================================================

# Unified wrapper first because later type references and
# conceptual dependency order rely on the wrapper existing.
output_parts.append(
    "\n\n# ============================================================\n"
    "# VERIFIED UNIFIED GROUNDING MODEL\n"
    "# ============================================================\n\n"
)

output_parts.append(
    unified_model_source
)


output_parts.append(
    "\n\n# ============================================================\n"
    "# VERIFIED INTEGRATION HELPERS\n"
    "# ============================================================\n\n"
)

for helper_name, helper_item in helper_items:

    output_parts.append(
        f"# Helper: {helper_name}\n"
    )

    output_parts.append(
        node_source(
            helper_item["lines"],
            helper_item["node"],
        )
    )

    output_parts.append("\n")


output_parts.append(
    "\n# ============================================================\n"
    "# VERIFIED UNIFIED GROUNDING INFERENCE\n"
    "# ============================================================\n\n"
)

output_parts.append(
    inference_source
)


output_parts.append(
    "\n\n# ============================================================\n"
    "# VERIFIED SATQUERYAI GROUNDING ROUTER\n"
    "# ============================================================\n\n"
)

output_parts.append(
    router_source
)


output = "".join(output_parts)


# ============================================================
# COMPILE BEFORE WRITING
# ============================================================

print("=" * 72)
print("COMPILING CLEAN RECOVERED MODULE")
print("=" * 72)

compile(
    output,
    str(
        OUT_DIR
        / "unified_grounding_recovered.py"
    ),
    "exec",
)

print("Compilation: PASS")
print()


# ============================================================
# WRITE OUTPUT
# ============================================================

output_path = (
    OUT_DIR
    / "unified_grounding_recovered.py"
)

output_path.write_text(
    output,
    encoding="utf-8",
)


# ============================================================
# PRESERVE EXACT SOURCE CELLS
# ============================================================

cell59_path = (
    OUT_DIR
    / "unified_model_cell_59_original.py"
)

cell59_source = selected_classes[
    "UnifiedGroundingModel"
]["source"]

cell59_path.write_text(
    cell59_source,
    encoding="utf-8",
)


cell66_path = (
    OUT_DIR
    / "integration_cell_66_original.py"
)

cell66_source = parsed_cells[
    PREFERRED_HELPER_CELL
]["source"]

cell66_path.write_text(
    cell66_source,
    encoding="utf-8",
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "notebook": str(NOTEBOOK),

    "selected_classes": {
        name: {
            "cell": item["cell"],
            "source_line":
                item["node"].lineno,
        }
        for name, item
        in selected_classes.items()
    },

    "selected_helpers": {
        name: {
            "cell": item["cell"],
            "source_line":
                item["node"].lineno,
        }
        for name, item
        in selected_helpers.items()
    },

    "required_helpers":
        sorted(REQUIRED_HELPERS),

    "selected_imports":
        filtered_imports,

    "future_imports":
        future_import_lines,

    "output":
        str(output_path),

    "excluded": [
        "notebook assignments",
        "dataset construction",
        "DataLoaders",
        "training code",
        "checkpoint loading",
        "evaluation objects",
        "temporary notebook state",
        "model instances",
        "Kaggle runtime state",
    ],
}


manifest_path = (
    OUT_DIR
    / "integration_recovery_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL REPORT
# ============================================================

print("=" * 72)
print("CLEAN INTEGRATION RECOVERY COMPLETE")
print("=" * 72)

print()
print("Selected classes:")

for name, item in selected_classes.items():
    print(
        f"  {name}: cell {item['cell']}"
    )

print()
print("Selected helpers:")

for name, item in helper_items:
    print(
        f"  {name}: "
        f"cell {item['cell']}, "
        f"line {item['node'].lineno}"
    )

print()
print(
    "Notebook assignments: EXCLUDED"
)

print(
    "Dataset construction: EXCLUDED"
)

print(
    "Training: EXCLUDED"
)

print(
    "Checkpoint loading: EXCLUDED"
)

print(
    "Kaggle: NOT USED"
)

print(
    "GPU: OFF"
)

print()
print(
    f"Output:\n  {output_path}"
)

print(
    f"Manifest:\n  {manifest_path}"
)

print()
print("=" * 72)