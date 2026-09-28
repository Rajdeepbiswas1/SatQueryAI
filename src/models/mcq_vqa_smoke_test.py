from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "src")

from models.mcq_vqa import MCQVQAModel


# ============================================================
# PATHS
# ============================================================

ANNOTATIONS = Path(
    r"data\processed\training_annotations.parquet"
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
TEST_BATCHES = 10
DEVICE = torch.device("cpu")


# ============================================================
# LOAD MCQ MAPPING
# ============================================================

print("Loading MCQ mapping...")

mapping = pd.read_parquet(
    MAPPING_FILE
)

print(
    "Mapping rows:",
    len(mapping)
)


# ============================================================
# LOAD QUESTION EMBEDDINGS
# ============================================================

print("\nLoading question embeddings...")

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
# LOAD DOFA SHARD INDEX
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
# PREPARE MCQ SAMPLES
# ============================================================

samples = []

for _, row in mapping.iterrows():

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
            f"Unexpected answer: {answer}"
        )

    embedding_index = int(
        row["question_embedding_index"]
    )

    patch_id = str(
        row["patch_id"]
    )

    if embedding_index not in question_embeddings:
        raise RuntimeError(
            f"Missing question embedding "
            f"index: {embedding_index}"
        )

    if patch_id not in patch_to_shard:
        raise RuntimeError(
            f"Missing DOFA patch: {patch_id}"
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
        }
    )

print(
    "\nPrepared MCQ samples:",
    len(samples)
)


# ============================================================
# DATASET
# ============================================================

class MCQSmokeDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[
            index
        ]

        patch_id = sample[
            "patch_id"
        ]

        embedding_index = sample[
            "embedding_index"
        ]

        shard_index = patch_to_shard[
            patch_id
        ]

        position = patch_to_position[
            patch_id
        ]

        # This smoke test uses a temporary
        # per-item shard load. Only 10 batches
        # are used, so this is acceptable.
        data = torch.load(
            dofa_files[shard_index],
            map_location="cpu",
            weights_only=False,
        )

        s1 = data["s1"][
            position
        ].float()

        s2 = data["s2"][
            position
        ].float()

        question = question_embeddings[
            embedding_index
        ]

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return (
            s1,
            s2,
            question,
            label,
        )


dataset = MCQSmokeDataset(
    samples
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)


# ============================================================
# MODEL
# ============================================================

model = MCQVQAModel().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# SMOKE TEST
# ============================================================

print("\nStarting real-data smoke test...")

model.train()

initial_parameters = {
    name: parameter.detach().clone()
    for name, parameter
    in model.named_parameters()
    if parameter.requires_grad
}

for batch_number, batch in enumerate(
    loader,
    start=1,
):

    if batch_number > TEST_BATCHES:
        break

    s1, s2, question, labels = batch

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

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Non-finite loss encountered."
        )

    loss.backward()

    max_gradient = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_value = (
                parameter.grad.abs().max().item()
            )

            max_gradient = max(
                max_gradient,
                gradient_value,
            )

    if max_gradient == 0.0:
        raise RuntimeError(
            "No gradients detected."
        )

    optimizer.step()

    predictions = logits.argmax(
        dim=1
    )

    accuracy = (
        predictions == labels
    ).float().mean().item()

    print(
        f"Batch {batch_number}/"
        f"{TEST_BATCHES} | "
        f"Loss {loss.item():.4f} | "
        f"Accuracy {accuracy:.4f} | "
        f"Max grad {max_gradient:.6f}"
    )


# ============================================================
# VERIFY PARAMETER UPDATE
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
        "Model parameters did not change "
        "after optimizer updates."
    )


print("\n" + "=" * 60)
print("MCQ REAL-DATA SMOKE TEST PASSED")
print("=" * 60)

print(
    "Real MCQ samples:",
    len(samples)
)

print(
    "Question embeddings:",
    len(question_embeddings)
)

print(
    "DOFA patches:",
    len(patch_to_shard)
)

print(
    "Output classes: 4 (A/B/C/D)"
)

print(
    "Gradients: detected"
)

print(
    "Parameters: updated"
)

print("=" * 60)