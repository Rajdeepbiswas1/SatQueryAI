from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from models.mcq_vqa_option_aware import MCQOptionAwareVQA


# ============================================================
# PATHS
# ============================================================

SPLIT_FILE = Path(
    r"data\processed\mcq_vqa_split.parquet"
)

MAPPING_FILE = Path(
    r"data\processed\mcq_question_embeddings"
    r"\mcq_row_mapping.parquet"
)

OPTION_CACHE = Path(
    r"data\processed\mcq_option_embeddings"
)

DOFA_CACHE = Path(
    r"data\processed\dofa_features_download"
    r"\dofa_features"
)

CHECKPOINT = Path(
    r"models\checkpoints\mcq_vqa_option_aware"
    r"\best.pt"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32
DEVICE = torch.device("cpu")

LABEL_TO_NAME = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


# ============================================================
# LOAD OPTION EMBEDDINGS
# ============================================================

print("Loading option-aware embeddings...")

option_embeddings = {}

option_files = sorted(
    OPTION_CACHE.glob("shard_*.pt")
)

if len(option_files) != 30:
    raise RuntimeError(
        f"Expected 30 option shards, "
        f"found {len(option_files)}"
    )

for file in option_files:

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    indices = data[
        "question_indices"
    ].tolist()

    q = data[
        "question_embeddings"
    ].float()

    a = data[
        "option_a_embeddings"
    ].float()

    b = data[
        "option_b_embeddings"
    ].float()

    c = data[
        "option_c_embeddings"
    ].float()

    d = data[
        "option_d_embeddings"
    ].float()

    for i, index in enumerate(indices):

        option_embeddings[int(index)] = {
            "question": q[i],
            "a": a[i],
            "b": b[i],
            "c": c[i],
            "d": d[i],
        }

print(
    "Cached questions:",
    len(option_embeddings)
)


# ============================================================
# LOAD MCQ SPLIT + MAPPING
# ============================================================

print("\nLoading validation data...")

split = pd.read_parquet(
    SPLIT_FILE
)

mapping = pd.read_parquet(
    MAPPING_FILE
)

mapping_lookup = {
    str(row["ID"]): int(
        row["question_embedding_index"]
    )
    for _, row in mapping.iterrows()
}

val = split[
    (split["type"] == "mcq")
    & (split["model_split"] == "val")
].copy()

val["question_embedding_index"] = (
    val["ID"]
    .astype(str)
    .map(mapping_lookup)
)

if val[
    "question_embedding_index"
].isna().any():

    raise RuntimeError(
        "Missing question embedding index."
    )

print(
    "Validation rows:",
    len(val)
)


# ============================================================
# INDEX DOFA
# ============================================================

print("\nIndexing DOFA shards...")

dofa_files = sorted(
    DOFA_CACHE.glob("shard_*.pt")
)

if len(dofa_files) != 40:
    raise RuntimeError(
        f"Expected 40 DOFA shards, "
        f"found {len(dofa_files)}"
    )

patch_to_shard = {}
patch_to_position = {}

for shard_index, file in enumerate(
    dofa_files
):

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    for position, patch_id in enumerate(
        data["patch_ids"]
    ):

        patch_id = str(patch_id)

        patch_to_shard[
            patch_id
        ] = shard_index

        patch_to_position[
            patch_id
        ] = position

print(
    "Indexed patches:",
    len(patch_to_shard)
)


# ============================================================
# GROUP VALIDATION ROWS BY SHARD
# ============================================================

groups = {
    shard: []
    for shard in range(
        len(dofa_files)
    )
}

label_map = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
}

for _, row in val.iterrows():

    answer = (
        str(row["output"])
        .strip()
        .lower()
    )

    if answer not in label_map:
        raise RuntimeError(
            f"Unexpected answer: {answer}"
        )

    patch_id = str(
        row["patch_id"]
    )

    embedding_index = int(
        row["question_embedding_index"]
    )

    if patch_id not in patch_to_shard:
        raise RuntimeError(
            f"Missing DOFA patch: {patch_id}"
        )

    if embedding_index not in option_embeddings:
        raise RuntimeError(
            f"Missing option embeddings: "
            f"{embedding_index}"
        )

    # Store everything needed for evaluation.
    sample = {
        "patch_id": patch_id,
        "embedding_index": embedding_index,
        "label": label_map[answer],
        "category": str(
            row["category"]
        ),
    }

    shard = patch_to_shard[
        patch_id
    ]

    groups[shard].append(
        sample
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

model = MCQOptionAwareVQA().to(
    DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Stored validation accuracy:",
    f"{checkpoint['val_accuracy'] * 100:.2f}%"
)


# ============================================================
# METRICS
# ============================================================

criterion = nn.CrossEntropyLoss()

total_loss = 0.0
total_correct = 0
total_examples = 0

confusion = torch.zeros(
    4,
    4,
    dtype=torch.long,
)

category_stats = {}


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating validation set...")

with torch.no_grad():

    for shard_number, shard in enumerate(
        sorted(
            groups.keys()
        ),
        start=1,
    ):

        shard_samples = groups[
            shard
        ]

        if not shard_samples:
            continue

        data = torch.load(
            dofa_files[shard],
            map_location="cpu",
            weights_only=False,
        )

        for start in range(
            0,
            len(shard_samples),
            BATCH_SIZE,
        ):

            batch = shard_samples[
                start:start + BATCH_SIZE
            ]

            s1_list = []
            s2_list = []
            question_list = []
            options_list = []
            labels_list = []
            categories = []

            for sample in batch:

                patch_id = sample[
                    "patch_id"
                ]

                embedding_index = sample[
                    "embedding_index"
                ]

                position = patch_to_position[
                    patch_id
                ]

                cached = option_embeddings[
                    embedding_index
                ]

                s1_list.append(
                    data["s1"][
                        position
                    ].float()
                )

                s2_list.append(
                    data["s2"][
                        position
                    ].float()
                )

                question_list.append(
                    cached["question"]
                )

                options_list.append(
                    torch.stack(
                        [
                            cached["a"],
                            cached["b"],
                            cached["c"],
                            cached["d"],
                        ]
                    )
                )

                labels_list.append(
                    sample["label"]
                )

                categories.append(
                    sample["category"]
                )

            s1 = torch.stack(
                s1_list
            )

            s2 = torch.stack(
                s2_list
            )

            questions = torch.stack(
                question_list
            )

            options = torch.stack(
                options_list
            )

            labels = torch.tensor(
                labels_list,
                dtype=torch.long,
            )

            logits = model(
                s1,
                s2,
                questions,
                options,
            )

            loss = criterion(
                logits,
                labels,
            )

            predictions = logits.argmax(
                dim=1
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

            # -----------------------------------------------
            # CONFUSION MATRIX
            # -----------------------------------------------

            for true, pred in zip(
                labels.tolist(),
                predictions.tolist(),
            ):

                confusion[
                    true,
                    pred
                ] += 1


            # -----------------------------------------------
            # CATEGORY STATS
            # -----------------------------------------------

            for true, pred, category in zip(
                labels.tolist(),
                predictions.tolist(),
                categories,
            ):

                if category not in category_stats:

                    category_stats[
                        category
                    ] = {
                        "correct": 0,
                        "total": 0,
                    }

                category_stats[
                    category
                ]["total"] += 1

                if true == pred:

                    category_stats[
                        category
                    ]["correct"] += 1

        del data


# ============================================================
# RESULTS
# ============================================================

average_loss = (
    total_loss
    / total_examples
)

accuracy = (
    total_correct
    / total_examples
)


print("\n" + "=" * 70)
print("OPTION-AWARE MCQ VALIDATION ANALYSIS")
print("=" * 70)

print(
    f"Examples:     {total_examples}"
)

print(
    f"Loss:         {average_loss:.4f}"
)

print(
    f"Accuracy:     {accuracy:.4f}"
)

print(
    f"Accuracy (%): {accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("-" * 70)

print(
    "Rows = actual, columns = predicted"
)

print(
    "          A       B       C       D"
)

for i in range(4):

    values = [
        int(confusion[i, j])
        for j in range(4)
    ]

    print(
        f"{LABEL_TO_NAME[i]}     "
        f"{values[0]:6d}"
        f"{values[1]:8d}"
        f"{values[2]:8d}"
        f"{values[3]:8d}"
    )


# ============================================================
# PER-CLASS RECALL
# ============================================================

print("\nPER-CLASS RECALL")
print("-" * 50)

for i in range(4):

    actual = confusion[
        i
    ].sum().item()

    correct = confusion[
        i,
        i
    ].item()

    recall = (
        correct / actual
        if actual
        else 0.0
    )

    print(
        f"{LABEL_TO_NAME[i]}: "
        f"{recall * 100:.2f}% "
        f"({correct}/{actual})"
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\nPREDICTION DISTRIBUTION")
print("-" * 50)

predicted_totals = (
    confusion.sum(dim=0)
)

for i in range(4):

    count = predicted_totals[
        i
    ].item()

    percentage = (
        count
        / total_examples
        * 100
    )

    print(
        f"{LABEL_TO_NAME[i]}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# CATEGORY ACCURACY
# ============================================================

print("\nCATEGORY ACCURACY")
print("-" * 50)

for category in sorted(
    category_stats
):

    correct = category_stats[
        category
    ]["correct"]

    total = category_stats[
        category
    ]["total"]

    category_accuracy = (
        correct / total
    )

    print(
        f"{category:20s} "
        f"{correct:5d}/"
        f"{total:<5d} "
        f"{category_accuracy * 100:.2f}%"
    )


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)