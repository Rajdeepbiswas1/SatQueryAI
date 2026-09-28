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
OVERFIT_EXAMPLES = 256
EPOCHS = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

SEED = 42

DEVICE = torch.device("cpu")

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD OPTION-AWARE EMBEDDINGS
# ============================================================

print("Loading option-aware embeddings...")

option_embeddings = {}

for file in sorted(
    OPTION_CACHE.glob("shard_*.pt")
):

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
# LOAD DATA + MAPPING
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
        "question embedding index."
    )

train = mcq[
    mcq["model_split"] == "train"
].copy()


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
            "patch_id": str(
                row["patch_id"]
            ),

            "embedding_index":
                embedding_index,

            "label":
                label_map[answer],
        }
    )


# Fixed subset
random.Random(
    SEED
).shuffle(samples)

samples = samples[
    :OVERFIT_EXAMPLES
]

print(
    "\nOverfit examples:",
    len(samples)
)


# ============================================================
# INDEX DOFA
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


# ============================================================
# LOAD FIXED OVERFIT SET
# ============================================================

print("\nLoading overfit features...")

# Keep each required DOFA shard in memory only once.
required_shards = {}

s1_list = []
s2_list = []
question_list = []
options_list = []
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
            options[start:end],
            labels[start:end],
        )
    )


# ============================================================
# OVERFIT
# ============================================================

print("\nStarting option-aware overfit test...")

start_time = time.perf_counter()

best_accuracy = 0.0

for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for (
        batch_s1,
        batch_s2,
        batch_questions,
        batch_options,
        batch_labels,
    ) in batches:

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

        if not torch.isfinite(loss):
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

        if max_gradient == 0.0:
            raise RuntimeError(
                "No gradients detected."
            )

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

    best_accuracy = max(
        best_accuracy,
        accuracy,
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
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("OPTION-AWARE MCQ OVERFIT DIAGNOSTIC")
print("=" * 60)

print(
    f"Examples tested: {len(samples)}"
)

print(
    f"Final accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Best accuracy:  "
    f"{best_accuracy * 100:.2f}%"
)

print("=" * 60)

if best_accuracy >= 0.90:

    print(
        "OVERFIT TEST PASSED"
    )

    print(
        "The option-aware model can "
        "memorize a small real MCQ set."
    )

else:

    print(
        "OVERFIT TEST DID NOT REACH 90%"
    )

    print(
        "The model learned some signal, "
        "but further diagnosis is needed."
    )