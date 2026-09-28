from pathlib import Path
import random
import sys
import time

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from models.feature_fusion import CrossModalFusion


# ============================================================
# CONFIG
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

CHECKPOINT_DIR = Path(
    r"models\checkpoints\binary_vqa"
)

RESUME_CHECKPOINT = CHECKPOINT_DIR / "last.pt"

BATCH_SIZE = 32

# IMPORTANT:
# Epoch 1 is already stored in last.pt.
# Setting this to 5 means we resume and continue through epoch 5.
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
# QUESTION EMBEDDINGS
# ============================================================

print("Loading question embeddings...")

question_ids = []
question_embeddings = []

question_files = sorted(
    QUESTION_CACHE.glob("shard_*.pt")
)

for file in question_files:

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    question_ids.extend(
        str(x) for x in data["ids"]
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
    tuple(question_embeddings.shape),
)

print(
    "Question lookup:",
    len(question_lookup),
)


# ============================================================
# ANNOTATIONS
# ============================================================

print("\nLoading annotations...")

annotations = pd.read_parquet(
    ANNOTATIONS
)

train_df = annotations[
    annotations["model_split"] == "train"
].copy()

val_df = annotations[
    annotations["model_split"] == "val"
].copy()


def prepare_samples(df):

    samples = []

    for _, row in df.iterrows():

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

    return samples


train_samples = prepare_samples(
    train_df
)

val_samples = prepare_samples(
    val_df
)

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

        patch_to_shard[patch_id] = (
            shard_index
        )

        patch_to_position[patch_id] = (
            position
        )

print(
    "DOFA shards:",
    len(dofa_files)
)

print(
    "Indexed patches:",
    len(patch_to_shard)
)


# ============================================================
# GROUP SAMPLES BY SHARD
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

class CachedBinaryVQA(nn.Module):

    def __init__(self):

        super().__init__()

        self.visual_fusion = CrossModalFusion(
            input_dim=768,
            hidden_dim=256,
        )

        self.question_projection = nn.Sequential(
            nn.Linear(
                768,
                256,
            ),
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
            nn.Linear(
                768,
                256,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                256,
                2,
            ),
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
            visual_summary
            * question
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


model = CachedBinaryVQA().to(
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
# CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RESUME FROM CHECKPOINT
# ============================================================

start_epoch = 1
best_val_accuracy = -1.0

if RESUME_CHECKPOINT.exists():

    print(
        f"\nLoading checkpoint: "
        f"{RESUME_CHECKPOINT}"
    )

    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    completed_epoch = int(
        checkpoint["epoch"]
    )

    start_epoch = (
        completed_epoch + 1
    )

    best_val_accuracy = float(
        checkpoint["val_accuracy"]
    )

    print(
        f"Resuming from epoch "
        f"{completed_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

else:

    print(
        "\nNo checkpoint found. "
        "Starting from epoch 1."
    )


# ============================================================
# PROCESS ONE SHARD AT A TIME
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
    Load one DOFA shard at a time.

    Samples inside the shard are randomized for training.
    The shard is then discarded before loading the next one.
    """

    file = dofa_files[
        shard_index
    ]

    data = torch.load(
        file,
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
        )

        s2 = torch.stack(
            s2_batch
        )

        question = torch.stack(
            question_batch
        )

        labels = torch.tensor(
            label_batch,
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

        current_batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * current_batch_size
        )

        total_correct += (
            predictions == labels
        ).sum().item()

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
# TRAINING
# ============================================================

if start_epoch > EPOCHS:

    print(
        "\nTraining is already complete "
        f"through epoch {EPOCHS}."
    )

else:

    print("\nStarting training...")

    print(
        "Epochs:",
        EPOCHS,
    )

    print(
        "Starting epoch:",
        start_epoch,
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

        # ====================================================
        # TRAIN
        # ====================================================

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
            train_correct / train_total
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for shard_position, shard in enumerate(
                [
                    s
                    for s, indices
                    in val_groups.items()
                    if indices
                ],
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
                    f"{len(val_groups)}"
                )

        val_loss /= val_total

        val_accuracy = (
            val_correct / val_total
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )


        # ====================================================
        # RESULTS
        # ====================================================

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


        # ====================================================
        # SAVE LAST CHECKPOINT
        # ====================================================

        last_checkpoint = (
            CHECKPOINT_DIR
            / "last.pt"
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
            last_checkpoint,
        )


        # ====================================================
        # SAVE BEST CHECKPOINT
        # ====================================================

        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            best_checkpoint = (
                CHECKPOINT_DIR
                / "best.pt"
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
                best_checkpoint,
            )

            print(
                "New best checkpoint saved."
            )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
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