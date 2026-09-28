from pathlib import Path
import random
import sys
import time

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from models.mcq_vqa import MCQVQAModel


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

QUESTION_CACHE = Path(
    r"data\processed\mcq_question_embeddings"
)

DOFA_CACHE = Path(
    r"data\processed\dofa_features_download"
    r"\dofa_features"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
OVERFIT_EXAMPLES = 256
EPOCHS = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

SEED = 42

DEVICE = torch.device("cpu")

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD QUESTION EMBEDDINGS
# ============================================================

print("Loading question embeddings...")

question_embeddings = {}

for file in sorted(
    QUESTION_CACHE.glob("shard_*.pt")
):

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    indices = data[
        "question_indices"
    ].tolist()

    embeddings = data[
        "question_embeddings"
    ].float()

    for i, index in enumerate(indices):

        question_embeddings[
            int(index)
        ] = embeddings[i]

print(
    "Cached question embeddings:",
    len(question_embeddings)
)


# ============================================================
# LOAD SPLIT + MAPPING
# ============================================================

print("\nLoading MCQ split...")

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
        "Some MCQ rows have no "
        "question embedding."
    )


train_df = mcq[
    mcq["model_split"] == "train"
].copy()


# ============================================================
# PREPARE SAMPLES
# ============================================================

samples = []

for _, row in train_df.iterrows():

    answer = (
        str(row["output"])
        .strip()
        .lower()
    )

    label_map = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
    }

    if answer not in label_map:
        raise RuntimeError(
            f"Unexpected answer: {answer}"
        )

    samples.append(
        {
            "patch_id": str(
                row["patch_id"]
            ),
            "embedding_index": int(
                row["question_embedding_index"]
            ),
            "label": label_map[answer],
        }
    )


# Shuffle once and take a fixed small subset.
rng = random.Random(SEED)

rng.shuffle(samples)

samples = samples[
    :OVERFIT_EXAMPLES
]

print(
    "\nOverfit examples:",
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
# LOAD THE SMALL OVERFIT SET
# ============================================================

print("\nLoading overfit examples...")

s1_list = []
s2_list = []
question_list = []
label_list = []

# Cache each required DOFA shard only once.
required_shards = {}

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
        question_embeddings[
            sample["embedding_index"]
        ]
    )

    label_list.append(
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

labels = torch.tensor(
    label_list,
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
    "Labels:",
    tuple(labels.shape)
)


# ============================================================
# MODEL
# ============================================================

model = MCQVQAModel().to(
    DEVICE
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# ============================================================
# FIXED BATCHES
# ============================================================

batches = []

for start in range(
    0,
    len(samples),
    BATCH_SIZE,
):

    end = min(
        start + BATCH_SIZE,
        len(samples),
    )

    batches.append(
        (
            s1[start:end],
            s2[start:end],
            questions[start:end],
            labels[start:end],
        )
    )


# ============================================================
# OVERFIT TEST
# ============================================================

print("\nStarting overfit test...")

start_time = time.perf_counter()

for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    # Always train on the exact same
    # examples in the exact same order.
    for (
        batch_s1,
        batch_s2,
        batch_questions,
        batch_labels,
    ) in batches:

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            batch_s1,
            batch_s2,
            batch_questions,
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

        optimizer.step()

        predictions = logits.argmax(
            dim=1
        )

        batch_size = (
            batch_labels.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_correct += (
            predictions
            == batch_labels
        ).sum().item()

        total += batch_size

    average_loss = (
        total_loss / total
    )

    accuracy = (
        total_correct / total
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss {average_loss:.4f} | "
        f"Accuracy {accuracy * 100:.2f}% | "
        f"Time {elapsed / 60:.2f} min"
    )


# ============================================================
# FINAL RESULT
# ============================================================

final_accuracy = accuracy

print("\n" + "=" * 60)
print("MCQ OVERFIT DIAGNOSTIC")
print("=" * 60)

print(
    f"Examples tested: {len(samples)}"
)

print(
    f"Final loss: {average_loss:.4f}"
)

print(
    f"Final accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print("=" * 60)

if final_accuracy >= 0.90:

    print(
        "OVERFIT TEST PASSED"
    )

    print(
        "The model can memorize a small "
        "real MCQ dataset."
    )

else:

    print(
        "OVERFIT TEST FAILED"
    )

    print(
        "The model did not reach 90% "
        "accuracy on the fixed subset."
    )