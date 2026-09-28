from pathlib import Path
import random
import sys

import pandas as pd
import torch

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

CHECKPOINT = Path(
    r"models\checkpoints\mcq_vqa\last.pt"
)


# ============================================================
# CONFIG
# ============================================================

NUM_SAMPLES = 1024
BATCH_SIZE = 32
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
    "Question embeddings:",
    len(question_embeddings)
)


# ============================================================
# LOAD MAPPING + VALIDATION SET
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


# Fixed random sample
val = (
    val.sample(
        n=min(NUM_SAMPLES, len(val)),
        random_state=SEED,
    )
    .reset_index(drop=True)
)

print(
    "Samples:",
    len(val)
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


# ============================================================
# LOAD SAMPLES
# ============================================================

print("\nLoading sample features...")

records = []

shard_cache = {}

for _, row in val.iterrows():

    patch_id = str(
        row["patch_id"]
    )

    shard_index = patch_to_shard[
        patch_id
    ]

    if shard_index not in shard_cache:

        shard_cache[
            shard_index
        ] = torch.load(
            dofa_files[shard_index],
            map_location="cpu",
            weights_only=False,
        )

    data = shard_cache[
        shard_index
    ]

    position = patch_to_position[
        patch_id
    ]

    embedding_index = int(
        row["question_embedding_index"]
    )

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

    records.append(
        {
            "s1": data["s1"][
                position
            ].float(),

            "s2": data["s2"][
                position
            ].float(),

            "question": question_embeddings[
                embedding_index
            ],

            "label": label_map[
                answer
            ],
        }
    )


# ============================================================
# FREE SHARDS
# ============================================================

del shard_cache


# ============================================================
# STACK DATA
# ============================================================

s1 = torch.stack(
    [
        x["s1"]
        for x in records
    ]
)

s2 = torch.stack(
    [
        x["s2"]
        for x in records
    ]
)

questions = torch.stack(
    [
        x["question"]
        for x in records
    ]
)

labels = torch.tensor(
    [
        x["label"]
        for x in records
    ],
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


# ============================================================
# SHUFFLED QUESTIONS
# ============================================================

shuffle_generator = torch.Generator()

shuffle_generator.manual_seed(
    SEED
)

shuffle_indices = torch.randperm(
    len(questions),
    generator=shuffle_generator,
)

shuffled_questions = questions[
    shuffle_indices
]


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading MCQ checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

model = MCQVQAModel().to(
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
    checkpoint["val_accuracy"]
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    questions_to_use,
):

    correct = 0
    total = 0

    with torch.no_grad():

        for start in range(
            0,
            len(labels),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(labels),
            )

            logits = model(
                s1[start:end],
                s2[start:end],
                questions_to_use[
                    start:end
                ],
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            correct += (
                predictions
                == labels[start:end]
            ).sum().item()

            total += (
                end - start
            )

    return correct / total


# ============================================================
# CORRECT QUESTION EVALUATION
# ============================================================

print("\nEvaluating correct questions...")

correct_accuracy = evaluate(
    questions
)


# ============================================================
# SHUFFLED QUESTION EVALUATION
# ============================================================

print(
    "Evaluating shuffled questions..."
)

shuffled_accuracy = evaluate(
    shuffled_questions
)


# ============================================================
# RESULTS
# ============================================================

difference = (
    correct_accuracy
    - shuffled_accuracy
)

print("\n" + "=" * 60)
print("MCQ QUESTION-SHUFFLING DIAGNOSTIC")
print("=" * 60)

print(
    f"Correct questions:  "
    f"{correct_accuracy * 100:.2f}%"
)

print(
    f"Shuffled questions: "
    f"{shuffled_accuracy * 100:.2f}%"
)

print(
    f"Accuracy difference: "
    f"{difference * 100:.2f} percentage points"
)

print("=" * 60)


if difference >= 0.10:

    print(
        "\nSTRONG QUESTION DEPENDENCE"
    )

    print(
        "The model is using the question "
        "information meaningfully."
    )

elif difference >= 0.03:

    print(
        "\nMODERATE QUESTION DEPENDENCE"
    )

    print(
        "The model uses question information, "
        "but the effect is limited."
    )

else:

    print(
        "\nWEAK QUESTION DEPENDENCE"
    )

    print(
        "The model's predictions change "
        "little when questions are shuffled."
    )