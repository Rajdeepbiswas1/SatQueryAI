from pathlib import Path
import json
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.cross_modal_mcq_vqa import CrossModalMCQVQA


# ============================================================
# CONFIG
# ============================================================

ROOT = Path.cwd()

ANNOTATIONS = (
    ROOT
    / "data"
    / "processed"
    / "mcq_vqa_split.parquet"
)

QUESTION_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "mcq_question_embeddings"
)

OPTION_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "mcq_option_embeddings"
)

DOFA_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

CHECKPOINT_DIR = (
    ROOT
    / "models"
    / "checkpoints"
    / "cross_modal_vqa"
    / "mcq"
)

BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last.pt"
HISTORY_FILE = CHECKPOINT_DIR / "history.csv"
CONFIG_FILE = CHECKPOINT_DIR / "config.json"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

INPUT_DIM = 768
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_OPTIONS = 4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("SATQUERYAI — CROSS-MODAL MCQ VQA TRAINING")
print("=" * 80)

print()
print("Project root:")
print(ROOT)

print()
print("Device:")
print(DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# PATH CHECKS
# ============================================================

required_paths = [
    ANNOTATIONS,
    QUESTION_CACHE,
    OPTION_CACHE,
    DOFA_CACHE,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("All required paths found.")


# ============================================================
# QUESTION CACHE
# ============================================================

print()
print("=" * 80)
print("LOADING QUESTION EMBEDDINGS")
print("=" * 80)


def find_ids(data):
    for key in (
        "question_indices",
        "ids",
        "question_ids",
    ):
        if key in data:
            return data[key]

    raise RuntimeError(
        "Could not find question-ID field. "
        f"Available keys: {sorted(data.keys())}"
    )


def find_question_embeddings(data):
    for key in (
        "question_embeddings",
        "embeddings",
        "question",
    ):
        if key in data:
            return data[key]

    raise RuntimeError(
        "Could not find question embedding field. "
        f"Available keys: {sorted(data.keys())}"
    )


def load_question_cache(cache_dir):

    lookup = {}

    files = sorted(
        cache_dir.glob("shard_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No question shards found in:\n{cache_dir}"
        )

    for file in files:

        data = torch.load(
            file,
            map_location="cpu",
            weights_only=False,
        )

        ids = find_ids(data)

        embeddings = find_question_embeddings(
            data
        )

        if not torch.is_tensor(embeddings):
            embeddings = torch.as_tensor(
                embeddings
            )

        embeddings = embeddings.float()

        if len(ids) != embeddings.shape[0]:
            raise RuntimeError(
                f"Question cache mismatch in {file.name}: "
                f"{len(ids)} IDs vs "
                f"{embeddings.shape[0]} embeddings"
            )

        for i, qid in enumerate(ids):
            lookup[int(qid)] = embeddings[i]

    print(
        "Question shards:",
        len(files)
    )

    print(
        "Question lookup:",
        len(lookup)
    )

    return lookup


question_lookup = load_question_cache(
    QUESTION_CACHE
)


# ============================================================
# OPTION CACHE
# ============================================================

print()
print("=" * 80)
print("LOADING OPTION EMBEDDINGS")
print("=" * 80)


def find_option_embeddings(data):

    required = [
        "option_a_embeddings",
        "option_b_embeddings",
        "option_c_embeddings",
        "option_d_embeddings",
    ]

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:
        raise RuntimeError(
            "Missing MCQ option embedding fields: "
            + ", ".join(missing)
            + f"\nAvailable keys: {sorted(data.keys())}"
        )

    return torch.stack(
        [
            data["option_a_embeddings"],
            data["option_b_embeddings"],
            data["option_c_embeddings"],
            data["option_d_embeddings"],
        ],
        dim=1,
    )


def load_option_cache(cache_dir):

    lookup = {}

    files = sorted(
        cache_dir.glob("shard_*.pt")
    )

    if not files:
        raise RuntimeError(
            f"No option shards found in:\n{cache_dir}"
        )

    for file in files:

        data = torch.load(
            file,
            map_location="cpu",
            weights_only=False,
        )

        ids = find_ids(data)

        options = find_option_embeddings(
            data
        )

        if not torch.is_tensor(options):
            options = torch.as_tensor(
                options
            )

        options = options.float()

        if options.ndim != 3:
            raise RuntimeError(
                f"Expected option embeddings "
                f"[N,5,768] in {file.name}, "
                f"got {tuple(options.shape)}"
            )

        if options.shape[1] != NUM_OPTIONS:
            raise RuntimeError(
                f"Expected {NUM_OPTIONS} options "
                f"in {file.name}, "
                f"got {options.shape[1]}"
            )

        if options.shape[2] != INPUT_DIM:
            raise RuntimeError(
                f"Expected option dimension "
                f"{INPUT_DIM} in {file.name}, "
                f"got {options.shape[2]}"
            )

        if len(ids) != options.shape[0]:
            raise RuntimeError(
                f"Option cache mismatch in {file.name}: "
                f"{len(ids)} IDs vs "
                f"{options.shape[0]} option rows"
            )

        for i, qid in enumerate(ids):
            lookup[int(qid)] = options[i]

    print(
        "Option shards:",
        len(files)
    )

    print(
        "Option lookup:",
        len(lookup)
    )

    return lookup


option_lookup = load_option_cache(
    OPTION_CACHE
)


# ============================================================
# MCQ DATA
# ============================================================

print()
print("=" * 80)
print("LOADING MCQ DATA")
print("=" * 80)

annotations = pd.read_parquet(
    ANNOTATIONS
)

print(
    "Total annotation rows:",
    len(annotations)
)

print(
    "Columns:",
    list(annotations.columns)
)


# ============================================================
# SPLIT COLUMN
# ============================================================

if "model_split" in annotations.columns:
    split_column = "model_split"
elif "split" in annotations.columns:
    split_column = "split"
else:
    raise RuntimeError(
        "Could not find model split column."
    )

print(
    "Split column:",
    split_column
)


# ============================================================
# ANSWER PARSER
# ============================================================

LABEL_MAP = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
}


def parse_answer(value):

    if value is None:
        return None

    text = str(value).strip().lower()

    if text in LABEL_MAP:
        return LABEL_MAP[text]

    patterns = [
        r"^\(([a-e])\)",
        r"^([a-e])[\.\:\-\)]",
        r"^option\s+([a-e])",
        r"^answer\s+([a-e])",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text
        )

        if match:
            return LABEL_MAP[
                match.group(1)
            ]

    return None


# ============================================================
# SAMPLE PREPARATION
# ============================================================

def prepare_samples(df):

    samples = []

    skipped_answers = 0
    missing_questions = 0

    for _, row in df.iterrows():

        patch_id = str(
            row["patch_id"]
        )

        question_index = int(
            row["question_embedding_index"]
        )

        label = parse_answer(
            row["output"]
        )

        if label is None:
            skipped_answers += 1
            continue

        if question_index not in question_lookup:
            missing_questions += 1
            continue

        if question_index not in option_lookup:
            raise RuntimeError(
                "Missing option embedding for "
                f"question index {question_index}"
            )

        samples.append(
            {
                "patch_id": patch_id,
                "question_index": question_index,
                "label": label,
                "category": str(
                    row["category"]
                ),
            }
        )

    print(
        "Prepared samples:",
        len(samples)
    )

    print(
        "Skipped invalid answers:",
        skipped_answers
    )

    print(
        "Missing question embeddings:",
        missing_questions
    )

    return samples


train_df = annotations[
    annotations[split_column]
    .astype(str)
    .str.lower()
    .eq("train")
].copy()

val_df = annotations[
    annotations[split_column]
    .astype(str)
    .str.lower()
    .eq("val")
].copy()

train_samples = prepare_samples(
    train_df
)

val_samples = prepare_samples(
    val_df
)

print()
print(
    "Train samples:",
    len(train_samples)
)

print(
    "Validation samples:",
    len(val_samples)
)


# ============================================================
# DOFA SHARD INDEX
# ============================================================

print()
print("=" * 80)
print("INDEXING DOFA SHARDS")
print("=" * 80)

dofa_files = sorted(
    DOFA_CACHE.glob("shard_*.pt")
)

if not dofa_files:
    raise RuntimeError(
        "No DOFA shards found."
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

    patch_ids = data.get(
        "patch_ids"
    )

    if patch_ids is None:
        raise RuntimeError(
            f"Missing patch_ids in {file.name}"
        )

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
        shard_index: []
        for shard_index in range(
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
                "Missing DOFA patch:\n"
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
# MODEL
# ============================================================

print()
print("=" * 80)
print("BUILDING CROSS-MODAL MCQ MODEL")
print("=" * 80)

model = CrossModalMCQVQA(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_options=NUM_OPTIONS,
).to(DEVICE)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Trainable parameters:",
    trainable_parameters
)


# ============================================================
# OPTIMIZER / LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# BATCH BUILDER
# ============================================================

def build_batch(
    sample_indices,
    samples,
    dofa_data,
):

    s1_batch = []
    s2_batch = []
    question_batch = []
    option_batch = []
    labels = []
    categories = []

    for sample_index in sample_indices:

        sample = samples[
            sample_index
        ]

        patch_id = sample[
            "patch_id"
        ]

        position = patch_to_position[
            patch_id
        ]

        s1_batch.append(
            dofa_data["s1"][
                position
            ].float()
        )

        s2_batch.append(
            dofa_data["s2"][
                position
            ].float()
        )

        question_batch.append(
            question_lookup[
                sample["question_index"]
            ]
        )

        option_batch.append(
            option_lookup[
                sample["question_index"]
            ]
        )

        labels.append(
            sample["label"]
        )

        categories.append(
            sample["category"]
        )

    return (
        torch.stack(s1_batch),
        torch.stack(s2_batch),
        torch.stack(question_batch),
        torch.stack(option_batch),
        torch.tensor(
            labels,
            dtype=torch.long,
        ),
        categories,
    )


# ============================================================
# RUN EPOCH
# ============================================================

def run_epoch(
    samples,
    groups,
    training,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    category_correct = {}
    category_total = {}

    shard_numbers = list(
        groups.keys()
    )

    if training:
        random.shuffle(
            shard_numbers
        )

    for shard_number in shard_numbers:

        sample_indices = groups[
            shard_number
        ]

        if not sample_indices:
            continue

        dofa_data = torch.load(
            dofa_files[
                shard_number
            ],
            map_location="cpu",
            weights_only=False,
        )

        local_indices = list(
            sample_indices
        )

        if training:
            random.shuffle(
                local_indices
            )

        for start in range(
            0,
            len(local_indices),
            BATCH_SIZE,
        ):

            batch_indices = local_indices[
                start:
                start + BATCH_SIZE
            ]

            (
                s1,
                s2,
                question,
                options,
                labels,
                categories,
            ) = build_batch(
                batch_indices,
                samples,
                dofa_data,
            )

            s1 = s1.to(
                DEVICE,
                non_blocking=True,
            )

            s2 = s2.to(
                DEVICE,
                non_blocking=True,
            )

            question = question.to(
                DEVICE,
                non_blocking=True,
            )

            options = options.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
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

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            else:
                with torch.no_grad():

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

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            batch_count = (
                labels.size(0)
            )

            batch_correct = (
                predictions == labels
            ).sum().item()

            total_loss += (
                loss.item()
                * batch_count
            )

            total_correct += (
                batch_correct
            )

            total_count += (
                batch_count
            )

            for category, pred, target in zip(
                categories,
                predictions.detach().cpu().tolist(),
                labels.detach().cpu().tolist(),
            ):

                category_total[
                    category
                ] = (
                    category_total.get(
                        category,
                        0,
                    )
                    + 1
                )

                if pred == target:
                    category_correct[
                        category
                    ] = (
                        category_correct.get(
                            category,
                            0,
                        )
                        + 1
                    )

        del dofa_data

    average_loss = (
        total_loss / total_count
    )

    accuracy = (
        total_correct
        / total_count
    )

    category_accuracy = {
        category: (
            category_correct.get(
                category,
                0,
            )
            / count
        )
        for category, count
        in category_total.items()
    }

    return (
        average_loss,
        accuracy,
        category_accuracy,
    )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)

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

print()

history = []

best_val_accuracy = -1.0

for epoch in range(
    1,
    EPOCHS + 1,
):

    epoch_start = time.time()

    train_loss, train_accuracy, _ = run_epoch(
        train_samples,
        train_groups,
        training=True,
    )

    val_loss, val_accuracy, val_categories = run_epoch(
        val_samples,
        val_groups,
        training=False,
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    print("=" * 80)
    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        f"Train loss:     {train_loss:.4f}"
    )

    print(
        f"Train accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Val loss:       {val_loss:.4f}"
    )

    print(
        f"Val accuracy:   {val_accuracy:.4f}"
    )

    print(
        f"Epoch time:     {epoch_time / 60:.2f} min"
    )

    if val_categories:

        print()
        print(
            "Validation category accuracy:"
        )

        for category in sorted(
            val_categories
        ):
            print(
                f"  {category}: "
                f"{val_categories[category]:.4f}"
            )

    print("=" * 80)

    history_row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "epoch_time_sec": epoch_time,
    }

    history.append(
        history_row
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "trainable_parameters":
            trainable_parameters,
        "config": {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS,
            "num_options": NUM_OPTIONS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
    }

    torch.save(
        checkpoint,
        LAST_CHECKPOINT,
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        torch.save(
            checkpoint,
            BEST_CHECKPOINT,
        )

        print(
            "New best checkpoint saved."
        )

    print()


# ============================================================
# SAVE HISTORY / CONFIG
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_FILE,
    index=False,
)

config = {
    "model": "CrossModalMCQVQA",
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_heads": NUM_HEADS,
    "num_options": NUM_OPTIONS,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "trainable_parameters":
        trainable_parameters,
    "best_validation_accuracy":
        best_val_accuracy,
}

with CONFIG_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
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

print(
    "Config:",
    CONFIG_FILE,
)