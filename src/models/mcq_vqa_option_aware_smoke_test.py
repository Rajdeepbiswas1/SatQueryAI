from pathlib import Path
import random
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
TEST_BATCHES = 10
DEVICE = torch.device("cpu")


# ============================================================
# LOAD OPTION-AWARE EMBEDDINGS
# ============================================================

print("Loading option-aware embeddings...")

option_embeddings = {}

option_files = sorted(
    OPTION_CACHE.glob("shard_*.pt")
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

    for i, index in enumerate(indices):

        option_embeddings[int(index)] = {
            "question": data[
                "question_embeddings"
            ][i].float(),

            "a": data[
                "option_a_embeddings"
            ][i].float(),

            "b": data[
                "option_b_embeddings"
            ][i].float(),

            "c": data[
                "option_c_embeddings"
            ][i].float(),

            "d": data[
                "option_d_embeddings"
            ][i].float(),
        }


print(
    "Cached MCQ questions:",
    len(option_embeddings)
)


# ============================================================
# LOAD SPLIT + MAPPING
# ============================================================

print("\nLoading MCQ data...")

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

mcq = split[
    split["type"] == "mcq"
].copy()

mcq["question_embedding_index"] = (
    mcq["ID"]
    .astype(str)
    .map(mapping_lookup)
)

if mcq[
    "question_embedding_index"
].isna().any():

    raise RuntimeError(
        "Some MCQ rows have no embedding index."
    )

train = mcq[
    mcq["model_split"] == "train"
].copy()

print(
    "Training rows:",
    len(train)
)


# ============================================================
# PREPARE SAMPLES
# ============================================================

label_map = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
}

samples = []

for _, row in train.iterrows():

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

    if embedding_index not in option_embeddings:
        raise RuntimeError(
            f"Missing option embeddings: "
            f"{embedding_index}"
        )

    samples.append(
        {
            "patch_id": patch_id,
            "embedding_index":
                embedding_index,
            "label":
                label_map[answer],
        }
    )


# Use only 10 batches for smoke testing.
random.Random(42).shuffle(
    samples
)

samples = samples[
    :BATCH_SIZE * TEST_BATCHES
]

print(
    "Smoke-test samples:",
    len(samples)
)


# ============================================================
# INDEX DOFA SHARDS
# ============================================================

print("\nIndexing DOFA shards...")

dofa_files = sorted(
    DOFA_CACHE.glob("shard_*.pt")
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
    "DOFA shards:",
    len(dofa_files)
)

print(
    "Indexed patches:",
    len(patch_to_shard)
)


# ============================================================
# LOAD REAL FEATURES
# ============================================================

print("\nLoading smoke-test features...")

required_shards = {}

s1_list = []
s2_list = []
question_list = []
option_list = []
labels_list = []

for sample in samples:

    patch_id = sample[
        "patch_id"
    ]

    shard_index = patch_to_shard[
        patch_id
    ]

    if shard_index not in required_shards:

        required_shards[
            shard_index
        ] = torch.load(
            dofa_files[shard_index],
            map_location="cpu",
            weights_only=False,
        )

    data = required_shards[
        shard_index
    ]

    position = patch_to_position[
        patch_id
    ]

    embedding_index = sample[
        "embedding_index"
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

    option_list.append(
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
    option_list
)

labels = torch.tensor(
    labels_list,
    dtype=torch.long,
)


print(
    "S1:",
    tuple(s1.shape)
)

print(
    "S2:",
    tuple(s2.shape)
)

print(
    "Questions:",
    tuple(questions.shape)
)

print(
    "Options:",
    tuple(options.shape)
)

print(
    "Labels:",
    tuple(labels.shape)
)


# ============================================================
# MODEL
# ============================================================

model = MCQOptionAwareVQA().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

criterion = nn.CrossEntropyLoss()


print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# ============================================================
# INITIAL PARAMETER SNAPSHOT
# ============================================================

initial_parameters = {
    name: parameter.detach().clone()
    for name, parameter
    in model.named_parameters()
    if parameter.requires_grad
}


# ============================================================
# TRAINING SMOKE TEST
# ============================================================

print("\nStarting real-data smoke test...")

model.train()

for batch_number, start in enumerate(
    range(
        0,
        len(samples),
        BATCH_SIZE,
    ),
    start=1,
):

    end = min(
        start + BATCH_SIZE,
        len(samples),
    )

    batch_s1 = s1[start:end]
    batch_s2 = s2[start:end]
    batch_questions = questions[start:end]
    batch_options = options[start:end]
    batch_labels = labels[start:end]

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        batch_s1,
        batch_s2,
        batch_questions,
        batch_options,
    )

    loss = criterion(
        logits,
        batch_labels,
    )

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Non-finite loss encountered."
        )

    loss.backward()

    max_gradient = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            max_gradient = max(
                max_gradient,
                parameter.grad.abs().max().item(),
            )

    if max_gradient == 0:

        raise RuntimeError(
            "No gradients detected."
        )

    optimizer.step()

    predictions = logits.argmax(
        dim=1
    )

    accuracy = (
        predictions == batch_labels
    ).float().mean().item()

    print(
        f"Batch {batch_number}/"
        f"{TEST_BATCHES} | "
        f"Loss {loss.item():.4f} | "
        f"Accuracy {accuracy:.4f} | "
        f"Max grad {max_gradient:.6f}"
    )


# ============================================================
# PARAMETER UPDATE CHECK
# ============================================================

updated = False

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if not torch.equal(
        initial_parameters[name],
        parameter.detach(),
    ):

        updated = True
        break


if not updated:
    raise RuntimeError(
        "Model parameters did not update."
    )


print("\n" + "=" * 60)
print("OPTION-AWARE MCQ REAL-DATA SMOKE TEST PASSED")
print("=" * 60)

print(
    "Real samples:",
    len(samples)
)

print(
    "Question embeddings:",
    len(option_embeddings)
)

print(
    "DOFA patches:",
    len(patch_to_shard)
)

print(
    "Output classes:",
    "A / B / C / D"
)

print(
    "Gradients: detected"
)

print(
    "Parameters: updated"
)

print("=" * 60)