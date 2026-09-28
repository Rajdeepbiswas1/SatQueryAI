from pathlib import Path
import random
import sys
import time

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from models.mcq_vqa_option_aware import MCQOptionAwareVQA


# ============================================================
# CONFIG
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

CHECKPOINT_DIR = Path(
    r"models\checkpoints\mcq_vqa_option_aware"
)

BATCH_SIZE = 32

# First proper run: 5 epochs
EPOCHS = 5

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
# LOAD OPTION-AWARE EMBEDDINGS
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

    question = data[
        "question_embeddings"
    ].float()

    option_a = data[
        "option_a_embeddings"
    ].float()

    option_b = data[
        "option_b_embeddings"
    ].float()

    option_c = data[
        "option_c_embeddings"
    ].float()

    option_d = data[
        "option_d_embeddings"
    ].float()

    rows = len(indices)

    expected_shape = (
        rows,
        768,
    )

    for name, tensor in [
        ("question", question),
        ("option_a", option_a),
        ("option_b", option_b),
        ("option_c", option_c),
        ("option_d", option_d),
    ]:

        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"{file.name}: {name} "
                f"has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}"
            )

    for i, index in enumerate(indices):

        option_embeddings[
            int(index)
        ] = {
            "question": question[i],
            "a": option_a[i],
            "b": option_b[i],
            "c": option_c[i],
            "d": option_d[i],
        }

print(
    "Cached MCQ questions:",
    len(option_embeddings)
)

if len(option_embeddings) != 60749:
    raise RuntimeError(
        f"Expected 60749 cached questions, "
        f"found {len(option_embeddings)}"
    )


# ============================================================
# LOAD SPLIT
# ============================================================

print("\nLoading MCQ split...")

split = pd.read_parquet(
    SPLIT_FILE
)

mapping = pd.read_parquet(
    MAPPING_FILE
)

print(
    "Mapping rows:",
    len(mapping)
)


# ============================================================
# QUESTION ID → EMBEDDING INDEX
# ============================================================

mapping_lookup = {
    str(row["ID"]): int(
        row["question_embedding_index"]
    )
    for _, row in mapping.iterrows()
}


# ============================================================
# SELECT MCQ ROWS
# ============================================================

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


train_df = mcq[
    mcq["model_split"] == "train"
].copy()

val_df = mcq[
    mcq["model_split"] == "val"
].copy()


# ============================================================
# SAMPLE PREPARATION
# ============================================================

label_map = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
}


def prepare_samples(df):

    samples = []

    for _, row in df.iterrows():

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

        if (
            embedding_index
            not in option_embeddings
        ):
            raise RuntimeError(
                f"Missing option embeddings "
                f"for index {embedding_index}"
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
                f"Missing DOFA patch: {patch_id}"
            )

        shard = patch_to_shard[
            patch_id
        ]

        groups[
            shard
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
# CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL
# ============================================================

model = MCQOptionAwareVQA().to(
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
# PROCESS ONE DOFA SHARD
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
    """
    Load one DOFA shard only.

    Shuffle examples inside the shard.
    Process all batches.
    Release the shard.
    """

    data = torch.load(
        dofa_files[shard_index],
        map_location="cpu",
        weights_only=False,
    )

    indices = list(
        sample_indices
    )

    if training:

        random.Random(
            epoch_seed + shard_index
        ).shuffle(indices)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

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
        options_batch = []
        labels_batch = []

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

            cached = option_embeddings[
                embedding_index
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
                cached["question"]
            )

            options_batch.append(
                torch.stack(
                    [
                        cached["a"],
                        cached["b"],
                        cached["c"],
                        cached["d"],
                    ]
                )
            )

            labels_batch.append(
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

        options = torch.stack(
            options_batch
        )

        labels = torch.tensor(
            labels_batch,
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
            options,
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

        total_examples += (
            batch_size
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

best_val_accuracy = -1.0

print("\nStarting training...")

print(
    "Epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)


for epoch in range(
    1,
    EPOCHS + 1,
):

    epoch_start = time.perf_counter()

    # --------------------------------------------------------
    # TRAINING
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
            f"Finished shard "
            f"{shard_position}/"
            f"{len(train_shards)} | "
            f"Train samples "
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

        for shard_position, shard in enumerate(
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
                f"{shard_position}/"
                f"{len(validation_shards)}"
            )

    val_loss /= val_total

    val_accuracy = (
        val_correct
        / val_total
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
    # LAST CHECKPOINT
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


    # --------------------------------------------------------
    # BEST CHECKPOINT
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

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
            },
            CHECKPOINT_DIR / "best.pt",
        )

        print(
            "New best checkpoint saved."
        )


print(
    "\n"
    + "=" * 70
)

print(
    "OPTION-AWARE MCQ TRAINING COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Checkpoints saved in: "
    f"{CHECKPOINT_DIR}"
)