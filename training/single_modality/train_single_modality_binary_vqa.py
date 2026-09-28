from pathlib import Path
import argparse
import random
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS = (
    ROOT
    / "data"
    / "processed"
    / "binary_vqa_split.parquet"
)

QUESTION_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "binary_question_embeddings"
)

DOFA_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "dofa_features_download"
    / "dofa_features"
)

CHECKPOINT_ROOT = (
    ROOT
    / "models"
    / "checkpoints"
    / "single_modality"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 128
EPOCHS = 8

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

VISUAL_DIM = 768
QUESTION_DIM = 768
HIDDEN_DIM = 256

NUM_WORKERS = 2

VAL_FRACTION = None


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# SINGLE-MODALITY MODEL
# ============================================================

class SingleModalityBinaryVQA(nn.Module):
    """
    Binary VQA model for ONE remote-sensing modality.

    Inputs
    ------
    visual_tokens:
        [B, 196, 768]

    question:
        [B, 768]

    Output
    ------
    logits:
        [B, 2]
    """

    def __init__(
        self,
        visual_dim=VISUAL_DIM,
        question_dim=QUESTION_DIM,
        hidden_dim=HIDDEN_DIM,
    ):
        super().__init__()

        self.visual_projection = nn.Sequential(
            nn.Linear(
                visual_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.question_projection = nn.Sequential(
            nn.Linear(
                question_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.question_query = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.visual_key = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.visual_value = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Dropout(0.10),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim,
                128,
            ),
            nn.GELU(),

            nn.Dropout(0.10),

            nn.Linear(
                128,
                2,
            ),
        )


    def forward(
        self,
        visual_tokens,
        question,
    ):

        visual = self.visual_projection(
            visual_tokens
        )

        question = self.question_projection(
            question
        )

        query = self.question_query(
            question
        ).unsqueeze(1)

        key = self.visual_key(
            visual
        )

        value = self.visual_value(
            visual
        )

        scores = (
            query * key
        ).sum(
            dim=-1
        ) / (
            HIDDEN_DIM ** 0.5
        )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attended_visual = (
            attention.unsqueeze(-1)
            * value
        ).sum(
            dim=1
        )

        joint = torch.cat(
            [
                attended_visual,
                question,
            ],
            dim=-1,
        )

        fused = self.fusion(
            joint
        )

        logits = self.classifier(
            fused
        )

        return logits


# ============================================================
# QUESTION CACHE
# ============================================================

def load_question_embeddings():

    print()
    print("=" * 70)
    print("LOADING QUESTION EMBEDDINGS")
    print("=" * 70)

    files = sorted(
        QUESTION_CACHE.glob(
            "shard_*.pt"
        )
    )

    if not files:
        raise RuntimeError(
            f"No question shards found:\n"
            f"{QUESTION_CACHE}"
        )

    question_ids = []
    question_embeddings = []

    for path in files:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            "ids" not in data
            or "question_embeddings" not in data
        ):
            raise RuntimeError(
                f"Unexpected question cache format:\n"
                f"{path}"
            )

        ids = [
            str(x)
            for x in data["ids"]
        ]

        embeddings = (
            data["question_embeddings"]
            .float()
        )

        question_ids.extend(
            ids
        )

        question_embeddings.append(
            embeddings
        )

    question_embeddings = torch.cat(
        question_embeddings,
        dim=0,
    )

    lookup = {
        qid: question_embeddings[i]
        for i, qid in enumerate(
            question_ids
        )
    }

    print(
        "Question shards:",
        len(files)
    )

    print(
        "Question embeddings:",
        tuple(
            question_embeddings.shape
        )
    )

    print(
        "Question IDs:",
        len(lookup)
    )

    return lookup


# ============================================================
# DOFA CACHE INDEX
# ============================================================

class DofaCache:

    def __init__(
        self,
        directory,
        modality,
    ):

        if modality not in {
            "s1",
            "s2",
        }:
            raise ValueError(
                "modality must be 's1' or 's2'"
            )

        self.directory = directory
        self.modality = modality

        self.files = sorted(
            directory.glob(
                "shard_*.pt"
            )
        )

        if not self.files:
            raise RuntimeError(
                f"No DOFA shards found:\n"
                f"{directory}"
            )

        self.index = {}
        self.loaded_shards = {}

        print()
        print("=" * 70)
        print(
            f"INDEXING DOFA CACHE — {modality.upper()}"
        )
        print("=" * 70)

        total = 0

        for shard_index, path in enumerate(
            self.files
        ):

            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            required = [
                "patch_ids",
                "s1",
                "s2",
            ]

            for key in required:
                if key not in data:
                    raise RuntimeError(
                        f"Missing key '{key}' in:\n"
                        f"{path}"
                    )

            patch_ids = data[
                "patch_ids"
            ]

            for row_index, patch_id in enumerate(
                patch_ids
            ):

                patch_id = str(
                    patch_id
                )

                if patch_id in self.index:
                    raise RuntimeError(
                        f"Duplicate patch ID:\n"
                        f"{patch_id}"
                    )

                self.index[
                    patch_id
                ] = (
                    shard_index,
                    row_index,
                )

                total += 1

        print(
            "Shards:",
            len(self.files)
        )

        print(
            "Indexed patch IDs:",
            total,
        )


    def _load_shard(
        self,
        shard_index,
    ):

        if (
            shard_index
            not in self.loaded_shards
        ):

            self.loaded_shards[
                shard_index
            ] = torch.load(
                self.files[
                    shard_index
                ],
                map_location="cpu",
                weights_only=False,
            )

        return self.loaded_shards[
            shard_index
        ]


    def get(
        self,
        patch_id,
    ):

        patch_id = str(
            patch_id
        )

        if patch_id not in self.index:
            raise KeyError(
                f"Patch not found in DOFA cache:\n"
                f"{patch_id}"
            )

        shard_index, row_index = (
            self.index[
                patch_id
            ]
        )

        shard = self._load_shard(
            shard_index
        )

        features = shard[
            self.modality
        ][
            row_index
        ].float()

        return features


# ============================================================
# DATASET
# ============================================================

class SingleModalityDataset(
    Dataset
):

    def __init__(
        self,
        samples,
        question_lookup,
        dofa_cache,
    ):

        self.samples = samples
        self.question_lookup = (
            question_lookup
        )
        self.dofa_cache = (
            dofa_cache
        )


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[
            index
        ]

        patch_id = (
            sample["patch_id"]
        )

        question_id = (
            sample["question_id"]
        )

        visual = self.dofa_cache.get(
            patch_id
        )

        question = (
            self.question_lookup[
                question_id
            ]
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return (
            visual,
            question,
            label,
        )


# ============================================================
# SAMPLE PREPARATION
# ============================================================

def prepare_samples(
    dataframe,
    question_lookup,
):

    samples = []

    for _, row in dataframe.iterrows():

        answer = (
            str(
                row["output"]
            )
            .strip()
            .lower()
        )

        if answer not in {
            "yes",
            "no",
        }:
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
                f"for question ID {question_id}"
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


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train,
):

    if train:
        model.train()
    else:
        model.eval()


    total_loss = 0.0
    correct = 0
    total = 0


    for visual, question, labels in loader:

        visual = visual.to(
            device,
            non_blocking=True,
        )

        question = question.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )


        if train:
            optimizer.zero_grad(
                set_to_none=True
            )


        with torch.set_grad_enabled(
            train
        ):

            logits = model(
                visual,
                question,
            )

            loss = criterion(
                logits,
                labels,
            )


            if train:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()


        batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        correct += int(
            (
                predictions
                == labels
            ).sum().item()
        )

        total += batch_size


    return (
        total_loss / total,
        correct / total,
    )


# ============================================================
# CATEGORY EVALUATION
# ============================================================

def evaluate_categories(
    model,
    dataframe,
    question_lookup,
    dofa_cache,
    device,
):

    model.eval()

    results = {}

    samples = prepare_samples(
        dataframe,
        question_lookup,
    )

    grouped = {}

    for sample in samples:

        grouped.setdefault(
            sample["category"],
            [],
        ).append(
            sample
        )

    with torch.no_grad():

        for category, group in grouped.items():

            correct = 0
            total = 0

            for sample in group:

                visual = (
                    dofa_cache.get(
                        sample["patch_id"]
                    )
                    .unsqueeze(0)
                    .to(device)
                )

                question = (
                    question_lookup[
                        sample["question_id"]
                    ]
                    .unsqueeze(0)
                    .to(device)
                )

                logits = model(
                    visual,
                    question,
                )

                prediction = int(
                    logits.argmax(
                        dim=1
                    ).item()
                )

                correct += (
                    prediction
                    == sample["label"]
                )

                total += 1

            results[
                category
            ] = (
                correct / total
            )

    return results


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
    modality,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
            "modality":
                modality,
            "architecture":
                "SingleModalityBinaryVQA",
            "visual_dim":
                VISUAL_DIM,
            "question_dim":
                QUESTION_DIM,
            "hidden_dim":
                HIDDEN_DIM,
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        required=True,
        choices=[
            "s1",
            "s2",
        ],
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    args = parser.parse_args()


    if args.device:
        device = torch.device(
            args.device
        )
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


    print()
    print("=" * 70)
    print("SATQUERYAI — SINGLE-MODALITY BINARY VQA")
    print("=" * 70)

    print(
        "Modality:",
        args.modality.upper()
    )

    print(
        "Device:",
        device
    )

    print(
        "Annotations:",
        ANNOTATIONS
    )

    print(
        "DOFA cache:",
        DOFA_CACHE
    )


    # --------------------------------------------------------
    # QUESTION EMBEDDINGS
    # --------------------------------------------------------

    question_lookup = (
        load_question_embeddings()
    )


    # --------------------------------------------------------
    # ANNOTATIONS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING ANNOTATIONS")
    print("=" * 70)

    annotations = pd.read_parquet(
        ANNOTATIONS
    )

    required_columns = {
        "model_split",
        "output",
        "ID",
        "patch_id",
        "category",
    }

    missing = (
        required_columns
        - set(
            annotations.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Missing annotation columns: "
            f"{sorted(missing)}"
        )

    train_df = annotations[
        annotations[
            "model_split"
        ] == "train"
    ].copy()

    val_df = annotations[
        annotations[
            "model_split"
        ] == "val"
    ].copy()

    print(
        "All annotation rows:",
        len(annotations),
    )

    print(
        "Train rows:",
        len(train_df),
    )

    print(
        "Validation rows:",
        len(val_df),
    )


    # --------------------------------------------------------
    # SAMPLES
    # --------------------------------------------------------

    train_samples = prepare_samples(
        train_df,
        question_lookup,
    )

    val_samples = prepare_samples(
        val_df,
        question_lookup,
    )

    print(
        "Usable train samples:",
        len(train_samples),
    )

    print(
        "Usable validation samples:",
        len(val_samples),
    )


    # --------------------------------------------------------
    # DOFA
    # --------------------------------------------------------

    dofa_cache = DofaCache(
        DOFA_CACHE,
        args.modality,
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    train_dataset = SingleModalityDataset(
        train_samples,
        question_lookup,
        dofa_cache,
    )

    val_dataset = SingleModalityDataset(
        val_samples,
        question_lookup,
        dofa_cache,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SingleModalityBinaryVQA()

    model.to(device)


    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir = (
        CHECKPOINT_ROOT
        / args.modality
        / "binary_vqa"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint = (
        output_dir
        / "best.pt"
    )

    last_checkpoint = (
        output_dir
        / "last.pt"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    best_val_accuracy = -1.0

    history = []


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        start_time = time.time()


        train_loss, train_accuracy = (
            run_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                train=True,
            )
        )


        val_loss, val_accuracy = (
            run_epoch(
                model,
                val_loader,
                optimizer=None,
                criterion=criterion,
                device=device,
                train=False,
            )
        )


        elapsed = (
            time.time()
            - start_time
        )


        print()
        print(
            "-" * 70
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.5f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val loss: "
            f"{val_loss:.5f}"
        )

        print(
            f"Val accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Epoch time: "
            f"{elapsed:.2f}s"
        )


        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            args.modality,
        )


        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy =
                val_accuracy

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                args.modality,
            )

            print(
                "NEW BEST CHECKPOINT"
            )


        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy":
                    train_accuracy,
                "val_loss": val_loss,
                "val_accuracy":
                    val_accuracy,
                "seconds":
                    elapsed,
            }
        )


    # --------------------------------------------------------
    # CATEGORY REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CATEGORY VALIDATION"
    )
    print("=" * 70)


    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    category_results = (
        evaluate_categories(
            model,
            val_df,
            question_lookup,
            dofa_cache,
            device,
        )
    )


    for category in sorted(
        category_results
    ):

        print(
            f"{category:20s}"
            f"{category_results[category] * 100:7.2f}%"
        )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_path = (
        output_dir
        / "history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )


    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Modality:",
        args.modality.upper()
    )

    print(
        "Best validation accuracy:",
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        "Best checkpoint:",
        best_checkpoint,
    )

    print(
        "History:",
        history_path,
    )


if __name__ == "__main__":
    main()