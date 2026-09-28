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
# CONFIG
# ============================================================

ANNOTATIONS = Path(
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

CHECKPOINT_DIR = Path(
    r"models\checkpoints\mcq_vqa"
)

BATCH_SIZE = 32
EPOCHS = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD QUESTION EMBEDDINGS
# ============================================================

print("Loading question embeddings...")

question_embeddings = {}

question_files = sorted(
    QUESTION_CACHE.glob("shard_*.pt")
)

for file in question_files:

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
# LOAD MCQ ROW MAPPING
# ============================================================

print("\nLoading MCQ mapping...")

mapping = pd.read_parquet(
    MAPPING_FILE
)

print(
    "Mapping rows:",
    len(mapping)
)


# ============================================================
# PREPARE TRAIN / VALIDATION SAMPLES
# ============================================================

# The original training_annotations.parquet contains
# the original rows. We use the verified row mapping only
# to obtain question_embedding_index.

mapping_lookup = {
    str(row["ID"]): int(
        row["question_embedding_index"]
    )
    for _, row in mapping.iterrows()
}


# Verify against original annotations
annotations = pd.read_parquet(
    ANNOTATIONS
)

mcq = annotations[
    annotations["type"] == "mcq"
].copy()


# Attach embedding index
mcq["question_embedding_index"] = (
    mcq["ID"]
    .astype(str)
    .map(mapping_lookup)
)


if mcq[
    "question_embedding_index"
].isna().any():

    missing = mcq[
        "question_embedding_index"
    ].isna().sum()

    raise RuntimeError(
        f"{missing} MCQ rows have no "
        f"question embedding index."
    )


# ------------------------------------------------------------
# IMPORTANT:
# The source annotation file doesn't have our model_split
# column guaranteed in every environment, so use the patch
# split if present.
# ------------------------------------------------------------

if "model_split" not in mcq.columns:

    raise RuntimeError(
        "model_split column not found in "
        "training_annotations.parquet."
    )


train_df = mcq[
    mcq["model_split"] == "train"
].copy()

val_df = mcq[
    mcq["model_split"] == "val"
].copy()


def prepare_samples(df):

    samples = []

    for _, row in df.iterrows():

        answer = (
            str(row["output"])
            .strip()
            .lower()
        )

        if answer not in {
            "a",
            "b",
            "c",
            "d",
        }:
            raise RuntimeError(
                f"Unexpected MCQ answer: "
                f"{answer}"
            )

        patch_id = str(
            row["patch_id"]
        )

        embedding_index = int(
            row["question_embedding_index"]
        )

        if embedding_index not in question_embeddings:
            raise RuntimeError(
                f"Missing question embedding: "
                f"{embedding_index}"
            )

        label = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
        }[answer]

        samples.append(
            {
                "patch_id": patch_id,
                "embedding_index":
                    embedding_index,
                "label": label,
                "category": str(
                    row["category"]
                ),
            }
        )

    return samples


train_samples = prepare_samples(
    train_df
)

val_samples = prepare_samples(
    val_df
)


print(
    "\nTrain samples:",
    len(train_samples)
)

print(
    "Validation samples:",
    len(val_samples)
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
# GROUP SAMPLES BY DOFA SHARD
# ============================================================

def group_samples(samples):

    groups = {
        shard: []
        for shard in range(
            len(dofa_files)
        )
    }

    for sample_index, sample in enumerate(
        samples
    ):

        patch_id = sample[
            "patch_id"
        ]

        if patch_id not in patch_to_shard:
            raise RuntimeError(
                f"Missing DOFA patch: "
                f"{patch_id}"
            )

        shard = patch_to_shard[
            patch_id
        ]

        groups[shard].append(
            sample_index
        )

    return groups


train_groups = group_samples(
    train_samples
)

val_groups = group_samples(
    val_samples
)


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL
# ============================================================

model = MCQVQAModel().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

criterion = nn.CrossEntropyLoss()

print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ),
)


# ============================================================
# PROCESS ONE SHARD
# ============================================================

def process_shard(
    model,
    optimizer,
    criterion,
    samples,
    sample_indices,
    shard_index,
    training,
    epoch_seed,
):

    # --------------------------------------------------------
    # Load exactly one shard
    # --------------------------------------------------------

    data = torch.load(
        dofa_files[shard_index],
        map_location="cpu",
        weights_only=False,
    )

    indices = list(
        sample_indices
    )

    rng = random.Random(
        epoch_seed + shard_index
    )

    if training:
        rng.shuffle(indices)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # --------------------------------------------------------
    # Batches
    # --------------------------------------------------------

    for start in range(
        0,
        len(indices),
        BATCH_SIZE,
    ):

        batch_indices = indices[
            start:start + BATCH_SIZE
        ]

        s1_batch = []
        s2_batch = []
        question_batch = []
        labels = []

        for sample_index in batch_indices:

            sample = samples[
                sample_index
            ]

            patch_id = sample[
                "patch_id"
            ]

            embedding_index = sample[
                "embedding_index"
            ]

            position = patch_to_position[
                patch_id
            ]

            s1_batch.append(
                data["s1"][
                    position
                ].float()
            )

            s2_batch.append(
                data["s2"][
                    position
                ].float()
            )

            question_batch.append(
                question_embeddings[
                    embedding_index
                ]
            )

            labels.append(
                sample["label"]
            )

        s1 = torch.stack(
            s1_batch
        )

        s2 = torch.stack(
            s2_batch
        )

        question = torch.stack(
            question_batch
        )

        labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        logits = model(
            s1,
            s2,
            question,
        )

        loss = criterion(
            logits,
            labels,
        )

        if training:

            loss.backward()
            optimizer.step()

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

    # Release shard from memory
    del data

    return (
        total_loss,
        total_correct,
        total_examples,
    )


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...")

for epoch in range(
    1,
    EPOCHS + 1,
):

    epoch_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_shards = [
        shard
        for shard, indices
        in train_groups.items()
        if indices
    ]

    random.Random(
        SEED + epoch
    ).shuffle(
        train_shards
    )

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for position, shard in enumerate(
        train_shards,
        start=1,
    ):

        (
            loss_sum,
            correct,
            count,
        ) = process_shard(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            samples=train_samples,
            sample_indices=train_groups[
                shard
            ],
            shard_index=shard,
            training=True,
            epoch_seed=SEED + epoch,
        )

        train_loss += loss_sum
        train_correct += correct
        train_total += count

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Finished shard "
            f"{position}/"
            f"{len(train_shards)} | "
            f"Train samples "
            f"{train_total}/"
            f"{len(train_samples)} | "
            f"Accuracy "
            f"{train_correct / train_total:.4f}"
        )

    train_loss /= train_total

    train_accuracy = (
        train_correct / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    validation_shards = [
        shard
        for shard, indices
        in val_groups.items()
        if indices
    ]

    with torch.no_grad():

        for position, shard in enumerate(
            validation_shards,
            start=1,
        ):

            (
                loss_sum,
                correct,
                count,
            ) = process_shard(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                samples=val_samples,
                sample_indices=val_groups[
                    shard
                ],
                shard_index=shard,
                training=False,
                epoch_seed=SEED,
            )

            val_loss += loss_sum
            val_correct += correct
            val_total += count

            print(
                f"Validation shard "
                f"{position}/"
                f"{len(validation_shards)}"
            )

    val_loss /= val_total

    val_accuracy = (
        val_correct / val_total
    )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"EPOCH {epoch}/{EPOCHS} COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Train loss:     "
        f"{train_loss:.4f}"
    )

    print(
        f"Train accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val loss:       "
        f"{val_loss:.4f}"
    )

    print(
        f"Val accuracy:   "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Epoch time:     "
        f"{epoch_time / 60:.2f} min"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "train_loss":
                train_loss,
            "train_accuracy":
                train_accuracy,
            "val_loss":
                val_loss,
            "val_accuracy":
                val_accuracy,
        },
        CHECKPOINT_DIR / "last.pt",
    )


print(
    "\n"
    + "=" * 70
)

print(
    "MCQ TRAINING COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Checkpoint: "
    f"{CHECKPOINT_DIR / 'last.pt'}"
)