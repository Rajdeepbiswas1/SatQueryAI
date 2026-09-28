from pathlib import Path
import csv
import json
import os
import random
import sys
import time

import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "SATQUERYAI_ROOT",
        Path.cwd(),
    )
)


# ============================================================
# PATHS
# ============================================================

ANNOTATIONS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_vqa_split.parquet"
)

QUESTION_CACHE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_question_embeddings"
)

DOFA_CACHE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "cross_modal_vqa"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "best.pt"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR / "last.pt"
)

HISTORY_FILE = (
    CHECKPOINT_DIR / "history.csv"
)

CONFIG_FILE = (
    CHECKPOINT_DIR / "config.json"
)


# ============================================================
# TRAINING CONFIG
# ============================================================

BATCH_SIZE = 32
EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# IMPORT MODEL
# ============================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.models.cross_modal_vqa import (
    CrossModalVQA,
)


# ============================================================
# HELPERS
# ============================================================

def require_path(
    path: Path,
    label: str,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


# ============================================================
# CHECK INPUTS
# ============================================================

print("=" * 80)
print("SATQUERYAI — CROSS-MODAL VQA TRAINING")
print("=" * 80)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Device:")
print(DEVICE)

require_path(
    ANNOTATIONS,
    "Binary VQA annotations",
)

require_path(
    QUESTION_CACHE,
    "Question embedding cache",
)

require_path(
    DOFA_CACHE,
    "DOFA cache",
)

print()
print("All required paths found.")


# ============================================================
# QUESTION EMBEDDINGS
# ============================================================

print()
print("=" * 80)
print("LOADING QUESTION EMBEDDINGS")
print("=" * 80)

question_files = sorted(
    QUESTION_CACHE.glob(
        "shard_*.pt"
    )
)

if not question_files:
    raise RuntimeError(
        "No question embedding shards found."
    )

question_ids = []
question_embeddings = []

for path in question_files:

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    question_ids.extend(
        str(x)
        for x in data["ids"]
    )

    question_embeddings.append(
        data[
            "question_embeddings"
        ].float()
    )

question_embeddings = torch.cat(
    question_embeddings,
    dim=0,
)

question_lookup = {
    question_id:
        question_embeddings[index]

    for index, question_id
    in enumerate(question_ids)
}

print(
    "Question shards:",
    len(question_files),
)

print(
    "Question embeddings:",
    tuple(
        question_embeddings.shape
    ),
)

print(
    "Question lookup:",
    len(question_lookup),
)


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print()
print("=" * 80)
print("LOADING BINARY VQA DATA")
print("=" * 80)

annotations = pd.read_parquet(
    ANNOTATIONS
)

train_df = annotations[
    annotations["model_split"]
    == "train"
].copy()

val_df = annotations[
    annotations["model_split"]
    == "val"
].copy()


def prepare_samples(df):

    samples = []

    for _, row in df.iterrows():

        answer = (
            str(row["output"])
            .strip()
            .lower()
        )

        if answer not in (
            "yes",
            "no",
        ):
            continue

        question_id = str(
            row["ID"]
        )

        patch_id = str(
            row["patch_id"]
        )

        if (
            question_id
            not in question_lookup
        ):
            raise RuntimeError(
                "Missing question embedding "
                f"for ID {question_id}"
            )

        samples.append(
            {
                "patch_id": patch_id,
                "question_id": question_id,
                "label": (
                    1
                    if answer == "yes"
                    else 0
                ),
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
    "Train samples:",
    len(train_samples),
)

print(
    "Validation samples:",
    len(val_samples),
)


# ============================================================
# DOFA SHARD INDEX
# ============================================================

print()
print("=" * 80)
print("INDEXING DOFA SHARDS")
print("=" * 80)

dofa_files = sorted(
    DOFA_CACHE.glob(
        "shard_*.pt"
    )
)

if not dofa_files:
    raise RuntimeError(
        "No DOFA feature shards found."
    )

patch_to_shard = {}
patch_to_position = {}

for shard_index, path in enumerate(
    dofa_files
):

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    patch_ids = data[
        "patch_ids"
    ]

    for position, patch_id in enumerate(
        patch_ids
    ):

        patch_id = str(
            patch_id
        )

        patch_to_shard[
            patch_id
        ] = shard_index

        patch_to_position[
            patch_id
        ] = position

print(
    "DOFA shards:",
    len(dofa_files),
)

print(
    "Indexed patches:",
    len(
        patch_to_shard
    ),
)


# ============================================================
# GROUP SAMPLES BY SHARD
# ============================================================

def group_samples(samples):

    groups = {
        shard_index: []
        for shard_index
        in range(
            len(dofa_files)
        )
    }

    for sample_index, sample in enumerate(
        samples
    ):

        patch_id = sample[
            "patch_id"
        ]

        if (
            patch_id
            not in patch_to_shard
        ):
            raise RuntimeError(
                "Missing DOFA patch:\n"
                f"{patch_id}"
            )

        shard_index = patch_to_shard[
            patch_id
        ]

        groups[
            shard_index
        ].append(
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
# MODEL
# ============================================================

print()
print("=" * 80)
print("BUILDING CROSS-MODAL MODEL")
print("=" * 80)

model = CrossModalVQA(
    input_dim=768,
    hidden_dim=256,
    num_heads=4,
).to(DEVICE)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}",
)


# ============================================================
# OPTIMIZER / LOSS
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG SAVE
# ============================================================

config = {
    "model": "CrossModalVQA",
    "input_dim": 768,
    "hidden_dim": 256,
    "num_heads": 4,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "device": str(DEVICE),
    "train_samples": len(
        train_samples
    ),
    "validation_samples": len(
        val_samples
    ),
}

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        config,
        file,
        indent=2,
    )


# ============================================================
# RESUME
# ============================================================

start_epoch = 1

best_val_accuracy = -1.0

if LAST_CHECKPOINT.exists():

    print()
    print(
        "Resuming from:",
        LAST_CHECKPOINT,
    )

    checkpoint = torch.load(
        LAST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    completed_epoch = int(
        checkpoint["epoch"]
    )

    start_epoch = (
        completed_epoch + 1
    )

    best_val_accuracy = float(
        checkpoint[
            "best_val_accuracy"
        ]
    )

    print(
        "Completed epoch:",
        completed_epoch,
    )

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy:.4f}",
    )


# ============================================================
# HISTORY FILE
# ============================================================

if not HISTORY_FILE.exists():

    with open(
        HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "epoch_minutes",
            ]
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

    path = dofa_files[
        shard_index
    ]

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    indices = list(
        sample_indices
    )

    rng = random.Random(
        epoch_seed
        + shard_index
    )

    if training:
        rng.shuffle(
            indices
        )

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for start in range(
        0,
        len(indices),
        BATCH_SIZE,
    ):

        batch_indices = indices[
            start:
            start + BATCH_SIZE
        ]

        s1_batch = []
        s2_batch = []
        question_batch = []
        label_batch = []

        for sample_index in batch_indices:

            sample = samples[
                sample_index
            ]

            patch_id = sample[
                "patch_id"
            ]

            question_id = sample[
                "question_id"
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
                question_lookup[
                    question_id
                ]
            )

            label_batch.append(
                sample["label"]
            )

        s1 = torch.stack(
            s1_batch
        ).to(
            DEVICE,
            non_blocking=True,
        )

        s2 = torch.stack(
            s2_batch
        ).to(
            DEVICE,
            non_blocking=True,
        )

        question = torch.stack(
            question_batch
        ).to(
            DEVICE,
            non_blocking=True,
        )

        labels = torch.tensor(
            label_batch,
            dtype=torch.long,
            device=DEVICE,
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

        current_batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * current_batch_size
        )

        total_correct += (
            (
                predictions
                == labels
            )
            .sum()
            .item()
        )

        total_examples += (
            current_batch_size
        )

    del data

    return (
        total_loss,
        total_correct,
        total_examples,
    )


# ============================================================
# TRAIN
# ============================================================

if start_epoch > EPOCHS:

    print()
    print(
        "Training already completed."
    )

else:

    print()
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    print(
        "Epochs:",
        EPOCHS,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

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

        for shard_position, shard in enumerate(
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
                f"Train shard "
                f"{shard_position}/"
                f"{len(train_shards)} | "
                f"Samples "
                f"{train_total}/"
                f"{len(train_samples)} | "
                f"Accuracy "
                f"{train_correct / train_total:.4f}"
            )

        train_loss /= train_total

        train_accuracy = (
            train_correct
            / train_total
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_shards = [
            shard
            for shard, indices
            in val_groups.items()
            if indices
        ]

        with torch.no_grad():

            for shard_position, shard in enumerate(
                val_shards,
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
                    f"{shard_position}/"
                    f"{len(val_shards)}"
                )

        val_loss /= val_total

        val_accuracy = (
            val_correct
            / val_total
        )

        epoch_minutes = (
            (
                time.perf_counter()
                - epoch_start
            )
            / 60.0
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 80)

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
            f"{epoch_minutes:.2f} min"
        )

        print("=" * 80)

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

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
                    "best_val_accuracy":
                        best_val_accuracy,
                    "config":
                        config,
                },
                BEST_CHECKPOINT,
            )

            print(
                "New best checkpoint saved."
            )

        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

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
                "best_val_accuracy":
                    best_val_accuracy,
                "config":
                    config,
            },
            LAST_CHECKPOINT,
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        with open(
            HISTORY_FILE,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_accuracy:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_accuracy:.6f}",
                    f"{epoch_minutes:.4f}",
                ]
            )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy:.4f}",
    )

    print(
        "Best checkpoint:",
        BEST_CHECKPOINT,
    )

    print(
        "Last checkpoint:",
        LAST_CHECKPOINT,
    )

    print(
        "History:",
        HISTORY_FILE,
    )