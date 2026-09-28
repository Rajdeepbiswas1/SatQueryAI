import os
import sys
from collections import Counter, defaultdict

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import torch
import numpy as np

from src.data.grounding_dataset import GroundingDataset
from src.models.grounding_heatmap_model import GroundingHeatmapModel


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    r"D:\SatQueryAI\models\checkpoints"
    r"\grounding_heatmap_geometry"
    r"\model_256_reference.pt"
)

START_INDEX = 256
EXAMPLES = 256

QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90

DEVICE = torch.device("cpu")


# ============================================================
# HELPERS
# ============================================================

def quantile_from_distribution(values, q):
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, 0.0)

    total = values.sum()

    if total <= 0:
        return 0.5

    values = values / total
    cdf = np.cumsum(values)

    index = np.searchsorted(
        cdf,
        q,
        side="left",
    )

    index = max(
        0,
        min(index, len(values) - 1),
    )

    return float(
        (index + 0.5) / len(values)
    )


def heatmap_to_box(heatmap):
    heatmap = np.asarray(
        heatmap,
        dtype=np.float64,
    )

    heatmap = np.maximum(
        heatmap,
        0.0,
    )

    total = heatmap.sum()

    if total <= 0:
        return [
            0.0,
            0.0,
            1.0,
            1.0,
        ]

    heatmap = heatmap / total

    x_distribution = heatmap.sum(axis=0)
    y_distribution = heatmap.sum(axis=1)

    x1 = quantile_from_distribution(
        x_distribution,
        QUANTILE_LOW,
    )

    x2 = quantile_from_distribution(
        x_distribution,
        QUANTILE_HIGH,
    )

    y1 = quantile_from_distribution(
        y_distribution,
        QUANTILE_LOW,
    )

    y2 = quantile_from_distribution(
        y_distribution,
        QUANTILE_HIGH,
    )

    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    ]


def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def extract_reference_phrase(row):
    """
    Try to recover the semantic reference phrase from the
    grounding row.

    Expected format may contain:
        <ref>...</ref>

    Falls back to input/output text if necessary.
    """

    candidates = []

    for column in [
        "input",
        "output",
        "question",
    ]:
        if column in row.index:
            value = row[column]

            if value is not None:
                candidates.append(
                    str(value)
                )

    for text in candidates:

        lower = text.lower()

        start = lower.find("<ref>")
        end = lower.find("</ref>")

        if start != -1 and end != -1:
            phrase = text[
                start + len("<ref>"):end
            ].strip()

            return phrase

    # No <ref> tags found.
    # Use the first available textual field.
    if candidates:
        return candidates[0]

    return ""


def get_category(row):
    if "category" in row.index:
        return clean_text(
            row["category"]
        )

    return "(unknown)"


def get_patch_id(row):
    if "patch_id" in row.index:
        return str(
            row["patch_id"]
        )

    if "ID" in row.index:
        return str(
            row["ID"]
        )

    return "(unknown)"


def load_checkpoint(model):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    current_state = model.state_dict()

    compatible_state = {}

    for key, value in state_dict.items():

        if key not in current_state:
            continue

        if (
            current_state[key].shape
            != value.shape
        ):
            continue

        compatible_state[key] = value

    model.load_state_dict(
        compatible_state,
        strict=False,
    )

    return len(compatible_state)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GROUNDING HELD-OUT DIAGNOSTICS")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    dataset = GroundingDataset()

    print(
        f"Total examples: {len(dataset):,}"
    )

    df = dataset.df

    end_index = (
        START_INDEX + EXAMPLES
    )

    assert end_index <= len(df)

    train_rows = df.iloc[
        0:START_INDEX
    ]

    heldout_rows = df.iloc[
        START_INDEX:end_index
    ]

    print(
        f"Training rows checked: "
        f"0 .. {START_INDEX - 1}"
    )

    print(
        f"Held-out rows checked: "
        f"{START_INDEX} .. {end_index - 1}"
    )

    # ========================================================
    # DIAGNOSTIC 1
    # QUERY/CATEGORY COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: ROWS 256-258")
    print("=" * 70)

    for idx in range(
        START_INDEX,
        START_INDEX + 3,
    ):

        row = df.iloc[idx]

        phrase = extract_reference_phrase(
            row
        )

        category = get_category(
            row
        )

        patch_id = get_patch_id(
            row
        )

        print(
            f"\nRow {idx}"
        )

        print(
            f"Patch ID: {patch_id}"
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Reference/query: {phrase}"
        )

    # Compare whether these three rows share
    # the same query/category.
    first_three = []

    for idx in range(
        START_INDEX,
        START_INDEX + 3,
    ):

        row = df.iloc[idx]

        first_three.append(
            (
                extract_reference_phrase(row),
                get_category(row),
            )
        )

    print("\nComparison:")

    if (
        len(set(first_three)) == 1
    ):
        print(
            "ALL THREE have the same "
            "reference phrase/category."
        )

    else:
        print(
            "The three rows do NOT all "
            "share the same phrase/category."
        )

    # ========================================================
    # DIAGNOSTIC 2
    # PATCH OVERLAP
    # ========================================================

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: PATCH OVERLAP")
    print("=" * 70)

    train_patch_ids = set(
        train_rows.apply(
            get_patch_id,
            axis=1,
        )
    )

    heldout_patch_ids = set(
        heldout_rows.apply(
            get_patch_id,
            axis=1,
        )
    )

    overlap = (
        train_patch_ids
        & heldout_patch_ids
    )

    print(
        f"Unique train patches in "
        f"rows 0-{START_INDEX - 1}: "
        f"{len(train_patch_ids):,}"
    )

    print(
        f"Unique held-out patches: "
        f"{len(heldout_patch_ids):,}"
    )

    print(
        f"Overlapping patch IDs: "
        f"{len(overlap):,}"
    )

    if len(overlap) == 0:

        print(
            "\nRESULT: No patch overlap."
        )

        print(
            "This 256-row experiment is "
            "patch-disjoint."
        )

    else:

        print(
            "\nRESULT: PATCH OVERLAP EXISTS."
        )

        print(
            "The held-out IoU may be "
            "inflated because some images "
            "also appeared in the training rows."
        )

        print("\nFirst overlapping patch IDs:")

        for patch_id in list(
            sorted(overlap)
        )[:20]:

            print(
                f"  {patch_id}"
            )

    # ========================================================
    # DIAGNOSTIC 3
    # PREDICTION DUPLICATES
    # ========================================================

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: PREDICTED-BOX DUPLICATES")
    print("=" * 70)

    model = GroundingHeatmapModel()

    loaded = load_checkpoint(
        model
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded compatible parameters: "
        f"{loaded}"
    )

    prediction_records = []

    print(
        "\nRunning predictions on "
        f"{EXAMPLES} held-out examples..."
    )

    with torch.no_grad():

        for local_index, dataset_index in enumerate(
            range(
                START_INDEX,
                end_index,
            )
        ):

            item = dataset[
                dataset_index
            ]

            s1 = item["s1"]
            s2 = item["s2"]
            text_embedding = item[
                "text_embedding"
            ]

            if s1.dim() == 2:
                s1 = s1.unsqueeze(0)

            if s2.dim() == 2:
                s2 = s2.unsqueeze(0)

            if text_embedding.dim() == 1:
                text_embedding = (
                    text_embedding.unsqueeze(0)
                )

            output = model(
                s1=s1.to(DEVICE),
                s2=s2.to(DEVICE),
                text_embedding=text_embedding.to(
                    DEVICE
                ),
            )

            heatmap = (
                output["heatmap"][0]
                .detach()
                .cpu()
                .numpy()
            )

            prediction = heatmap_to_box(
                heatmap
            )

            # Round to 3 decimals to identify
            # effectively identical predictions.
            rounded = tuple(
                round(
                    float(x),
                    3,
                )
                for x in prediction
            )

            row = df.iloc[
                dataset_index
            ]

            phrase = extract_reference_phrase(
                row
            )

            category = get_category(
                row
            )

            patch_id = get_patch_id(
                row
            )

            prediction_records.append(
                {
                    "index":
                        dataset_index,
                    "patch_id":
                        patch_id,
                    "phrase":
                        phrase,
                    "category":
                        category,
                    "prediction":
                        prediction,
                    "rounded":
                        rounded,
                }
            )

            if (
                local_index + 1
            ) % 32 == 0:

                print(
                    f"Processed "
                    f"{local_index + 1}/"
                    f"{EXAMPLES}"
                )

    # ========================================================
    # DUPLICATE ANALYSIS
    # ========================================================

    rounded_boxes = [
        record["rounded"]
        for record in prediction_records
    ]

    counts = Counter(
        rounded_boxes
    )

    unique_count = len(counts)

    duplicate_examples = (
        EXAMPLES - unique_count
    )

    print(
        f"\nUnique predicted boxes: "
        f"{unique_count}/{EXAMPLES}"
    )

    print(
        f"Duplicate assignments beyond "
        f"the first occurrence: "
        f"{duplicate_examples}"
    )

    duplicate_groups = [
        (
            box,
            count,
        )
        for box, count in counts.items()
        if count > 1
    ]

    duplicate_groups.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    print(
        f"Number of duplicated box groups: "
        f"{len(duplicate_groups)}"
    )

    if duplicate_groups:

        print(
            "\nTop duplicate prediction groups:"
        )

        for box, count in duplicate_groups[
            :10
        ]:

            print(
                f"\nBox {box} "
                f"→ {count} examples"
            )

            matching = [
                record
                for record in prediction_records
                if record["rounded"] == box
            ]

            for record in matching[:10]:

                print(
                    f"  index={record['index']} "
                    f"category={record['category']} "
                    f"patch={record['patch_id']}"
                )

                print(
                    f"    query={record['phrase']}"
                )

    else:

        print(
            "\nNo duplicate predicted boxes "
            "at 3-decimal precision."
        )

    # ========================================================
    # CATEGORY DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 70)
    print("HELD-OUT CATEGORY DISTRIBUTION")
    print("=" * 70)

    category_counts = Counter(
        record["category"]
        for record in prediction_records
    )

    for category, count in (
        category_counts
        .most_common()
    ):

        print(
            f"{category}: {count}"
        )

    # ========================================================
    # PHRASE DUPLICATION
    # ========================================================

    print("\n" + "=" * 70)
    print("HELD-OUT REFERENCE-PHRASE DISTRIBUTION")
    print("=" * 70)

    phrase_counts = Counter(
        record["phrase"]
        for record in prediction_records
    )

    print(
        f"Unique phrases in held-out rows: "
        f"{len(phrase_counts)}"
    )

    print(
        "\nMost common phrases:"
    )

    for phrase, count in (
        phrase_counts
        .most_common(15)
    ):

        print(
            f"{count:3d} × {phrase}"
        )

    # ========================================================
    # SAME QUERY → DIFFERENT IMAGES ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAME-QUERY IMAGE DIVERSITY")
    print("=" * 70)

    phrase_groups = defaultdict(list)

    for record in prediction_records:

        phrase_groups[
            record["phrase"]
        ].append(record)

    multi_image_phrases = []

    for phrase, records in (
        phrase_groups.items()
    ):

        patch_ids = set(
            record["patch_id"]
            for record in records
        )

        if len(patch_ids) >= 2:
            multi_image_phrases.append(
                (
                    phrase,
                    records,
                )
            )

    print(
        f"Queries occurring on multiple "
        f"images: "
        f"{len(multi_image_phrases)}"
    )

    # Show a few examples where the same query
    # occurs across different images.
    shown = 0

    for phrase, records in multi_image_phrases:

        print(
            f"\nQuery: {phrase}"
        )

        for record in records[:5]:

            print(
                f"  index={record['index']} "
                f"patch={record['patch_id']} "
                f"box={record['rounded']}"
            )

        shown += 1

        if shown >= 5:
            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"\n1. Rows 256-258 same "
        f"phrase/category: "
        f"{len(set(first_three)) == 1}"
    )

    print(
        f"2. Patch overlap: "
        f"{len(overlap)}"
    )

    print(
        f"3. Unique predictions: "
        f"{unique_count}/{EXAMPLES}"
    )

    print(
        f"4. Duplicate prediction groups: "
        f"{len(duplicate_groups)}"
    )

    print(
        "\nDiagnostics complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()