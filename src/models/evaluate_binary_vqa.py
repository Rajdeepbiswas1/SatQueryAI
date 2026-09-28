from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from models.feature_fusion import CrossModalFusion


# ============================================================
# PATHS
# ============================================================

ANNOTATIONS = Path(
    r"data\processed\binary_vqa_split.parquet"
)

QUESTION_CACHE = Path(
    r"data\processed\binary_question_embeddings"
)

DOFA_CACHE = Path(
    r"data\processed\dofa_features_download\dofa_features"
)

CHECKPOINT = Path(
    r"models\checkpoints\binary_vqa\best.pt"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32
DEVICE = torch.device("cpu")


# ============================================================
# LOAD QUESTION EMBEDDINGS
# ============================================================

print("Loading question embeddings...")

question_ids = []
question_embeddings = []

for file in sorted(
    QUESTION_CACHE.glob("shard_*.pt")
):

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    question_ids.extend(
        str(x)
        for x in data["ids"]
    )

    question_embeddings.append(
        data["question_embeddings"].float()
    )

question_embeddings = torch.cat(
    question_embeddings,
    dim=0,
)

question_lookup = {
    qid: question_embeddings[i]
    for i, qid in enumerate(question_ids)
}

print(
    "Question embeddings:",
    tuple(question_embeddings.shape)
)


# ============================================================
# LOAD VALIDATION ANNOTATIONS
# ============================================================

print("\nLoading validation annotations...")

annotations = pd.read_parquet(
    ANNOTATIONS
)

val_df = annotations[
    annotations["model_split"] == "val"
].copy()


# ============================================================
# PREPARE VALIDATION SAMPLES
# ============================================================

samples = []

for _, row in val_df.iterrows():

    answer = (
        str(row["output"])
        .strip()
        .lower()
    )

    if answer not in ("yes", "no"):
        continue

    qid = str(row["ID"])
    patch_id = str(row["patch_id"])

    if qid not in question_lookup:
        raise RuntimeError(
            f"Missing question embedding: {qid}"
        )

    samples.append(
        {
            "patch_id": patch_id,
            "question_id": qid,
            "label": 1 if answer == "yes" else 0,
            "category": str(row["category"]),
        }
    )

print(
    "Validation samples:",
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
# GROUP VALIDATION SAMPLES BY SHARD
# ============================================================

groups = {
    shard: []
    for shard in range(
        len(dofa_files)
    )
}

for sample_index, sample in enumerate(
    samples
):

    patch_id = sample["patch_id"]

    if patch_id not in patch_to_shard:
        raise RuntimeError(
            f"Missing DOFA patch: {patch_id}"
        )

    shard = patch_to_shard[
        patch_id
    ]

    groups[shard].append(
        sample_index
    )


# ============================================================
# MODEL
# ============================================================

class CachedBinaryVQA(nn.Module):

    def __init__(self):

        super().__init__()

        self.visual_fusion = CrossModalFusion(
            input_dim=768,
            hidden_dim=256,
        )

        self.question_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )

        self.visual_key = nn.Linear(
            256,
            256,
        )

        self.question_query = nn.Linear(
            256,
            256,
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        s1,
        s2,
        question,
    ):

        visual = self.visual_fusion(
            s1,
            s2,
        )

        question = self.question_projection(
            question
        )

        keys = self.visual_key(
            visual
        )

        query = self.question_query(
            question
        )

        scores = torch.bmm(
            keys,
            query.unsqueeze(-1),
        ).squeeze(-1)

        scores = (
            scores
            / (256 ** 0.5)
        )

        attention = torch.softmax(
            scores,
            dim=1,
        )

        visual_summary = torch.bmm(
            attention.unsqueeze(1),
            visual,
        ).squeeze(1)

        interaction = (
            visual_summary * question
        )

        combined = torch.cat(
            [
                visual_summary,
                question,
                interaction,
            ],
            dim=1,
        )

        return self.classifier(
            combined
        )


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print(
    "\nLoading best checkpoint..."
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

model = CachedBinaryVQA().to(
    DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
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
# EVALUATION
# ============================================================

criterion = nn.CrossEntropyLoss()

total_loss = 0.0
total_correct = 0
total_examples = 0

confusion = {
    "no_no": 0,
    "no_yes": 0,
    "yes_no": 0,
    "yes_yes": 0,
}

category_stats = {}


with torch.no_grad():

    for shard_index in range(
        len(dofa_files)
    ):

        sample_indices = groups[
            shard_index
        ]

        if not sample_indices:
            continue

        print(
            f"Evaluating shard "
            f"{shard_index + 1}/"
            f"{len(dofa_files)}"
        )

        data = torch.load(
            dofa_files[shard_index],
            map_location="cpu",
            weights_only=False,
        )

        for start in range(
            0,
            len(sample_indices),
            BATCH_SIZE,
        ):

            batch_indices = sample_indices[
                start:start + BATCH_SIZE
            ]

            s1_batch = []
            s2_batch = []
            question_batch = []
            labels = []
            categories = []

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

                labels.append(
                    sample["label"]
                )

                categories.append(
                    sample["category"]
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

            logits = model(
                s1,
                s2,
                question,
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

            total_examples += (
                batch_size
            )

            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            for true, pred in zip(
                labels.tolist(),
                predictions.tolist(),
            ):

                if true == 0 and pred == 0:
                    confusion["no_no"] += 1

                elif true == 0 and pred == 1:
                    confusion["no_yes"] += 1

                elif true == 1 and pred == 0:
                    confusion["yes_no"] += 1

                elif true == 1 and pred == 1:
                    confusion["yes_yes"] += 1


            # ------------------------------------------------
            # CATEGORY STATISTICS
            # ------------------------------------------------

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
    total_loss / total_examples
)

accuracy = (
    total_correct / total_examples
)


print("\n" + "=" * 70)
print("BINARY VQA VALIDATION ANALYSIS")
print("=" * 70)

print(
    f"Examples:        {total_examples}"
)

print(
    f"Loss:            {average_loss:.4f}"
)

print(
    f"Accuracy:        {accuracy:.4f}"
)

print(
    f"Accuracy (%):    {accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("-" * 40)

print(
    "                 Predicted NO   Predicted YES"
)

print(
    f"Actual NO       "
    f"{confusion['no_no']:13d}"
    f"{confusion['no_yes']:16d}"
)

print(
    f"Actual YES      "
    f"{confusion['yes_no']:13d}"
    f"{confusion['yes_yes']:16d}"
)


# ============================================================
# CLASS METRICS
# ============================================================

actual_no = (
    confusion["no_no"]
    + confusion["no_yes"]
)

actual_yes = (
    confusion["yes_no"]
    + confusion["yes_yes"]
)

no_recall = (
    confusion["no_no"] / actual_no
    if actual_no
    else 0
)

yes_recall = (
    confusion["yes_yes"] / actual_yes
    if actual_yes
    else 0
)

print("\nCLASS RECALL")
print("-" * 40)

print(
    f"NO recall:  "
    f"{no_recall:.4f} "
    f"({no_recall * 100:.2f}%)"
)

print(
    f"YES recall: "
    f"{yes_recall:.4f} "
    f"({yes_recall * 100:.2f}%)"
)


# ============================================================
# CATEGORY ACCURACY
# ============================================================

print("\nCATEGORY ACCURACY")
print("-" * 40)

for category in sorted(
    category_stats
):

    correct = category_stats[
        category
    ]["correct"]

    total = category_stats[
        category
    ]["total"]

    acc = correct / total

    print(
        f"{category:15s} "
        f"{correct:5d}/"
        f"{total:<5d} "
        f"{acc * 100:.2f}%"
    )


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)