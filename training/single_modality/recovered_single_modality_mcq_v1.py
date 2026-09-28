# Extracted from Kaggle notebook: notebook8dfff136e6(2).ipynb
# Cells 4 and 6 combined so the training definitions/data preparation are preserved together.
# This is a historical/reproducibility training script, not the runtime inference path.
# Kaggle paths are intentionally preserved for now.

# ============================================================
# SOURCE CELL 4
# ============================================================

from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "/kaggle/input/datasets/rajdeepbiswas1/"
    "satqueryai-single-modality-training"
)

DOFA_DIR = ROOT / "dofa_features"

BINARY_DIR = ROOT / "binary_vqa"
BINARY_PARQUET = BINARY_DIR / "binary_vqa_split.parquet"
BINARY_Q_DIR = BINARY_DIR / "question_embeddings"

MCQ_DIR = ROOT / "mcq_vqa"
MCQ_PARQUET = MCQ_DIR / "mcq_vqa_split.parquet"
MCQ_Q_DIR = MCQ_DIR / "question_embeddings"
MCQ_O_DIR = MCQ_DIR / "option_embeddings"

OUTPUT_ROOT = Path("/kaggle/working/satqueryai_single_modality_models")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

# IMPORTANT: fixed at 10 epochs for all six specialists.
NUM_EPOCHS = 10

BATCH_SIZE = 128
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("SATQUERYAI — SINGLE-MODALITY MODEL SANITY BUILD")
print("=" * 70)

print("Device:", DEVICE)
print("Epochs:", NUM_EPOCHS)
print("Batch size:", BATCH_SIZE)

# ============================================================
# DOFA CACHE
# ============================================================

print("\nLoading DOFA shards...")

dofa_patch_ids = []
dofa_s1 = []
dofa_s2 = []

dofa_files = sorted(DOFA_DIR.glob("*.pt"))

for i, path in enumerate(dofa_files):
    d = torch.load(path, map_location="cpu")

    dofa_patch_ids.append(
        [str(x) for x in d["patch_ids"]]
    )

    dofa_s1.append(
        d["s1"].contiguous()
    )

    dofa_s2.append(
        d["s2"].contiguous()
    )

dofa_patch_ids = sum(dofa_patch_ids, [])
dofa_s1 = torch.cat(dofa_s1, dim=0)
dofa_s2 = torch.cat(dofa_s2, dim=0)

assert len(dofa_patch_ids) == dofa_s1.shape[0]
assert len(dofa_patch_ids) == dofa_s2.shape[0]
assert len(set(dofa_patch_ids)) == len(dofa_patch_ids)

patch_to_index = {
    patch_id: i
    for i, patch_id in enumerate(dofa_patch_ids)
}

print("DOFA patches:", len(patch_to_index))
print("S1 tensor:", tuple(dofa_s1.shape), dofa_s1.dtype)
print("S2 tensor:", tuple(dofa_s2.shape), dofa_s2.dtype)

# ============================================================
# BINARY QUESTION CACHE
# ============================================================

print("\nLoading Binary question cache...")

binary_q_ids = []
binary_q_embeddings = []

for path in sorted(BINARY_Q_DIR.glob("*.pt")):
    d = torch.load(path, map_location="cpu")

    binary_q_ids.append(
        torch.as_tensor(
            d["ids"],
            dtype=torch.long
        )
    )

    binary_q_embeddings.append(
        d["question_embeddings"].float()
    )

binary_q_ids = torch.cat(binary_q_ids)
binary_q_embeddings = torch.cat(
    binary_q_embeddings,
    dim=0
)

binary_q_lookup = {
    int(qid): i
    for i, qid in enumerate(binary_q_ids.tolist())
}

assert len(binary_q_lookup) == len(binary_q_embeddings)

print("Binary question embeddings:",
      tuple(binary_q_embeddings.shape))

# ============================================================
# BINARY DATASET
# ============================================================

binary_df = pd.read_parquet(BINARY_PARQUET)

print("\nBinary rows:", len(binary_df))

binary_patch_indices = []
binary_question_indices = []
binary_labels = []

for row in binary_df.itertuples(index=False):

    patch_id = str(row.patch_id)
    qid = int(row.ID)
    answer = str(row.output).strip().lower()

    if patch_id not in patch_to_index:
        raise RuntimeError(
            f"Missing DOFA patch: {patch_id}"
        )

    if qid not in binary_q_lookup:
        raise RuntimeError(
            f"Missing question embedding ID: {qid}"
        )

    if answer not in {"yes", "no"}:
        raise RuntimeError(
            f"Unexpected Binary answer: {answer}"
        )

    binary_patch_indices.append(
        patch_to_index[patch_id]
    )

    binary_question_indices.append(
        binary_q_lookup[qid]
    )

    binary_labels.append(
        1 if answer == "yes" else 0
    )

binary_patch_indices = torch.tensor(
    binary_patch_indices,
    dtype=torch.long
)

binary_question_indices = torch.tensor(
    binary_question_indices,
    dtype=torch.long
)

binary_labels = torch.tensor(
    binary_labels,
    dtype=torch.long
)

# Keep the original train/val split.
binary_train_mask = (
    binary_df["model_split"].astype(str).values
    == "train"
)

binary_val_mask = (
    binary_df["model_split"].astype(str).values
    == "val"
)

class SingleBinaryDataset(Dataset):

    def __init__(
        self,
        patch_indices,
        question_indices,
        labels,
        mask,
        s_features,
        question_embeddings,
    ):
        mask = torch.as_tensor(
            mask,
            dtype=torch.bool
        )

        self.patch_indices = patch_indices[mask]
        self.question_indices = question_indices[mask]
        self.labels = labels[mask]

        self.s_features = s_features
        self.question_embeddings = question_embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        p = self.patch_indices[idx]
        q = self.question_indices[idx]

        return (
            self.s_features[p],
            self.question_embeddings[q],
            self.labels[idx],
        )

# ============================================================
# MCQ QUESTION + OPTION CACHE
# ============================================================

print("\nLoading MCQ question cache...")

mcq_q_files = sorted(
    MCQ_Q_DIR.glob("*.pt")
)

# The cache indices are 0..60748.
mcq_q_indices = []
mcq_q_values = []

for path in mcq_q_files:

    d = torch.load(
        path,
        map_location="cpu"
    )

    mcq_q_indices.append(
        d["question_indices"].long()
    )

    mcq_q_values.append(
        d["question_embeddings"].float()
    )

mcq_q_indices = torch.cat(
    mcq_q_indices
)

mcq_q_values = torch.cat(
    mcq_q_values,
    dim=0
)

mcq_num_questions = (
    int(mcq_q_indices.max().item()) + 1
)

mcq_questions = torch.empty(
    mcq_num_questions,
    768,
    dtype=torch.float32
)

mcq_questions[
    mcq_q_indices
] = mcq_q_values

assert mcq_questions.shape[0] == 60749

print(
    "MCQ question matrix:",
    tuple(mcq_questions.shape)
)

print("\nLoading MCQ option cache...")

mcq_o_files = sorted(
    MCQ_O_DIR.glob("*.pt")
)

mcq_o_indices = []
mcq_a = []
mcq_b = []
mcq_c = []
mcq_d = []

for path in mcq_o_files:

    d = torch.load(
        path,
        map_location="cpu"
    )

    mcq_o_indices.append(
        d["question_indices"].long()
    )

    mcq_a.append(
        d["option_a_embeddings"].float()
    )

    mcq_b.append(
        d["option_b_embeddings"].float()
    )

    mcq_c.append(
        d["option_c_embeddings"].float()
    )

    mcq_d.append(
        d["option_d_embeddings"].float()
    )

mcq_o_indices = torch.cat(mcq_o_indices)

mcq_options = torch.empty(
    60749,
    4,
    768,
    dtype=torch.float32
)

mcq_options[
    mcq_o_indices,
    0
] = torch.cat(mcq_a, dim=0)

mcq_options[
    mcq_o_indices,
    1
] = torch.cat(mcq_b, dim=0)

mcq_options[
    mcq_o_indices,
    2
] = torch.cat(mcq_c, dim=0)

mcq_options[
    mcq_o_indices,
    3
] = torch.cat(mcq_d, dim=0)

assert mcq_options.shape == (
    60749,
    4,
    768
)

print(
    "MCQ option matrix:",
    tuple(mcq_options.shape)
)

# ============================================================
# MCQ ROW → QUESTION INDEX
# ============================================================

mcq_df = pd.read_parquet(
    MCQ_PARQUET
)

mapping_df = pd.read_parquet(
    MCQ_Q_DIR / "mcq_row_mapping.parquet"
)

assert len(mcq_df) == len(mapping_df)

# Match by ID rather than trusting row order.
mapping_lookup = dict(
    zip(
        mapping_df["ID"].astype(int),
        mapping_df["question_embedding_index"].astype(int),
    )
)

mcq_patch_indices = []
mcq_question_indices = []
mcq_labels = []

for row in mcq_df.itertuples(index=False):

    patch_id = str(row.patch_id)
    qid = int(row.ID)

    if patch_id not in patch_to_index:
        raise RuntimeError(
            f"Missing MCQ DOFA patch: {patch_id}"
        )

    if qid not in mapping_lookup:
        raise RuntimeError(
            f"Missing MCQ question mapping: {qid}"
        )

    answer = str(row.output).strip().lower()

    if answer not in {"a", "b", "c", "d"}:
        raise RuntimeError(
            f"Unexpected MCQ answer: {answer}"
        )

    mcq_patch_indices.append(
        patch_to_index[patch_id]
    )

    mcq_question_indices.append(
        mapping_lookup[qid]
    )

    mcq_labels.append(
        {"a": 0, "b": 1, "c": 2, "d": 3}[answer]
    )

mcq_patch_indices = torch.tensor(
    mcq_patch_indices,
    dtype=torch.long
)

mcq_question_indices = torch.tensor(
    mcq_question_indices,
    dtype=torch.long
)

mcq_labels = torch.tensor(
    mcq_labels,
    dtype=torch.long
)

mcq_train_mask = (
    mcq_df["model_split"].astype(str).values
    == "train"
)

mcq_val_mask = (
    mcq_df["model_split"].astype(str).values
    == "val"
)

class SingleMCQDataset(Dataset):

    def __init__(
        self,
        patch_indices,
        question_indices,
        labels,
        mask,
        s_features,
        question_embeddings,
        option_embeddings,
    ):

        mask = torch.as_tensor(
            mask,
            dtype=torch.bool
        )

        self.patch_indices = patch_indices[mask]
        self.question_indices = question_indices[mask]
        self.labels = labels[mask]

        self.s_features = s_features
        self.question_embeddings = question_embeddings
        self.option_embeddings = option_embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        p = self.patch_indices[idx]
        q = self.question_indices[idx]

        return (
            self.s_features[p],
            self.question_embeddings[q],
            self.option_embeddings[q],
            self.labels[idx],
        )

# ============================================================
# TRUE SINGLE-MODALITY VQA MODEL
# ============================================================

class SingleModalityEncoder(nn.Module):

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        dropout=0.10,
    ):
        super().__init__()

        self.visual_proj = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.question_proj = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual,
        question,
    ):

        visual = self.visual_proj(
            visual.float()
        )

        question = self.question_proj(
            question.float()
        )

        visual_norm = F.normalize(
            visual,
            dim=-1
        )

        question_norm = F.normalize(
            question,
            dim=-1
        )

        attention_logits = torch.einsum(
            "bnd,bd->bn",
            visual_norm,
            question_norm,
        )

        attention = F.softmax(
            attention_logits * 16.0,
            dim=-1
        )

        context = torch.sum(
            visual * attention.unsqueeze(-1),
            dim=1
        )

        fused = self.fusion(
            torch.cat(
                [context, question],
                dim=-1
            )
        )

        return fused


class SingleModalityBinaryVQA(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = SingleModalityEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        visual,
        question,
    ):

        x = self.encoder(
            visual,
            question
        )

        return self.classifier(x)


class SingleModalityOptionAwareMCQ(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = SingleModalityEncoder()

        self.option_proj = nn.Linear(
            768,
            256
        )

        self.option_scorer = nn.Sequential(
            nn.Linear(
                256 * 3,
                256
            ),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.10),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        visual,
        question,
        options,
    ):

        context = self.encoder(
            visual,
            question
        )

        question_proj = self.encoder.question_proj(
            question.float()
        )

        option_proj = self.option_proj(
            options.float()
        )

        context = context.unsqueeze(1).expand(
            -1,
            4,
            -1
        )

        question_proj = question_proj.unsqueeze(1).expand(
            -1,
            4,
            -1
        )

        combined = torch.cat(
            [
                context,
                question_proj,
                option_proj,
            ],
            dim=-1
        )

        scores = self.option_scorer(
            combined
        ).squeeze(-1)

        return scores

# ============================================================
# CHECK FOUR MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODEL FORWARD/BACKWARD SANITY")
print("=" * 70)

def check_binary(
    visual_features,
    modality_name,
):

    ds = SingleBinaryDataset(
        binary_patch_indices,
        binary_question_indices,
        binary_labels,
        binary_train_mask,
        visual_features,
        binary_q_embeddings,
    )

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    visual, question, labels = next(
        iter(loader)
    )

    model = SingleModalityBinaryVQA().to(
        DEVICE
    )

    visual = visual.to(
        DEVICE,
        non_blocking=True
    )

    question = question.to(
        DEVICE,
        non_blocking=True
    )

    labels = labels.to(
        DEVICE,
        non_blocking=True
    )

    logits = model(
        visual,
        question
    )

    loss = F.cross_entropy(
        logits,
        labels
    )

    loss.backward()

    print(
        f"{modality_name} Binary:",
        "PASS",
        "| logits",
        tuple(logits.shape),
        "| loss",
        float(loss.detach().cpu())
    )


def check_mcq(
    visual_features,
    modality_name,
):

    ds = SingleMCQDataset(
        mcq_patch_indices,
        mcq_question_indices,
        mcq_labels,
        mcq_train_mask,
        visual_features,
        mcq_questions,
        mcq_options,
    )

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    visual, question, options, labels = next(
        iter(loader)
    )

    model = SingleModalityOptionAwareMCQ().to(
        DEVICE
    )

    visual = visual.to(DEVICE)
    question = question.to(DEVICE)
    options = options.to(DEVICE)
    labels = labels.to(DEVICE)

    logits = model(
        visual,
        question,
        options
    )

    loss = F.cross_entropy(
        logits,
        labels
    )

    loss.backward()

    print(
        f"{modality_name} MCQ:",
        "PASS",
        "| logits",
        tuple(logits.shape),
        "| loss",
        float(loss.detach().cpu())
    )


check_binary(dofa_s1, "S1")
check_binary(dofa_s2, "S2")

check_mcq(dofa_s1, "S1")
check_mcq(dofa_s2, "S2")

print("\n" + "=" * 70)
print("SINGLE-MODALITY SANITY BUILD: PASS")
print("=" * 70)

print(
    "\nReady for 10-epoch training:"
)
print("  S1 Binary VQA")
print("  S2 Binary VQA")
print("  S1 Option-Aware MCQ")
print("  S2 Option-Aware MCQ")

# ============================================================
# SOURCE CELL 6
# ============================================================

# ============================================================
# SATQUERYAI — TRAIN S1/S2 SINGLE-MODALITY OPTION-AWARE MCQ
# EXACTLY 10 EPOCHS
# ============================================================

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2
PIN_MEMORY = True

TRAIN_ROOT = Path(
    "/kaggle/working/satqueryai_single_modality_models"
)

TRAIN_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Dataset / loaders
# ------------------------------------------------------------

def make_mcq_loaders(visual_features):

    train_dataset = SingleMCQDataset(
        mcq_patch_indices,
        mcq_question_indices,
        mcq_labels,
        mcq_train_mask,
        visual_features,
        mcq_questions,
        mcq_options,
    )

    val_dataset = SingleMCQDataset(
        mcq_patch_indices,
        mcq_question_indices,
        mcq_labels,
        mcq_val_mask,
        visual_features,
        mcq_questions,
        mcq_options,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    )

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_mcq(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for (
        visual,
        question,
        options,
        labels,
    ) in loader:

        visual = visual.to(
            DEVICE,
            non_blocking=True
        )

        question = question.to(
            DEVICE,
            non_blocking=True
        )

        options = options.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda"),
        ):

            logits = model(
                visual,
                question,
                options
            )

            loss = F.cross_entropy(
                logits,
                labels
            )

        predictions = logits.argmax(
            dim=1
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_mcq_specialist(
    modality_name,
    visual_features,
):

    print("\n")
    print("=" * 70)
    print(f"TRAINING {modality_name} OPTION-AWARE MCQ")
    print("=" * 70)

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = make_mcq_loaders(
        visual_features
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    model = (
        SingleModalityOptionAwareMCQ()
        .to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.10,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    checkpoint_dir = (
        TRAIN_ROOT
        / modality_name.lower()
        / "mcq_vqa"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_checkpoint = (
        checkpoint_dir
        / "best.pt"
    )

    last_checkpoint = (
        checkpoint_dir
        / "last.pt"
    )

    history_path = (
        checkpoint_dir
        / "training_history.json"
    )

    history = []

    best_val_accuracy = -1.0
    best_val_loss = float("inf")

    total_start = time.time()

    print("\nStarting training...")
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Device:", DEVICE)

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for (
            visual,
            question,
            options,
            labels,
        ) in train_loader:

            visual = visual.to(
                DEVICE,
                non_blocking=True
            )

            question = question.to(
                DEVICE,
                non_blocking=True
            )

            options = options.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):

                logits = model(
                    visual,
                    question,
                    options
                )

                loss = F.cross_entropy(
                    logits,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            predictions = logits.argmax(
                dim=1
            )

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            running_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            running_samples += batch_size

        scheduler.step()

        train_loss = (
            running_loss
            / running_samples
        )

        train_accuracy = (
            running_correct
            / running_samples
        )

        validation = evaluate_mcq(
            model,
            val_loader
        )

        val_loss = validation["loss"]
        val_accuracy = validation["accuracy"]

        epoch_time = (
            time.time()
            - epoch_start
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }

        history.append(record)

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "modality": modality_name,
                "model_type": (
                    "single_modality_option_aware_mcq_vqa"
                ),
                "config": {
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "input_dim": 768,
                    "hidden_dim": 256,
                    "num_options": 4,
                },
                "metrics": record,
            },
            last_checkpoint,
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        is_better = (
            val_accuracy > best_val_accuracy
            or (
                val_accuracy == best_val_accuracy
                and val_loss < best_val_loss
            )
        )

        if is_better:

            best_val_accuracy = val_accuracy
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "modality": modality_name,
                    "model_type": (
                        "single_modality_option_aware_mcq_vqa"
                    ),
                    "config": {
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "input_dim": 768,
                        "hidden_dim": 256,
                        "num_options": 4,
                    },
                    "metrics": record,
                },
                best_checkpoint,
            )

            best_marker = "  <-- NEW BEST"

        else:
            best_marker = ""

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_accuracy:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_accuracy:.4f} | "
            f"{epoch_time:.1f}s"
            f"{best_marker}"
        )

    total_time = (
        time.time()
        - total_start
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    best_record = max(
        history,
        key=lambda x: (
            x["val_accuracy"],
            -x["val_loss"],
        )
    )

    history_payload = {
        "modality": modality_name,
        "model_type": (
            "single_modality_option_aware_mcq_vqa"
        ),
        "epochs_completed": EPOCHS,
        "best_epoch": best_record["epoch"],
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "total_time_sec": total_time,
        "history": history,
    }

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history_payload,
            f,
            indent=2
        )

    print("\nTraining complete:")
    print("Modality:", modality_name)
    print("Epochs completed:", EPOCHS)
    print(
        "Best validation accuracy:",
        f"{best_val_accuracy:.4f}"
    )
    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )
    print(
        "Best checkpoint:",
        best_checkpoint
    )

    return history_payload


# ============================================================
# TRAIN S1 MCQ
# ============================================================

s1_mcq_result = train_mcq_specialist(
    modality_name="S1",
    visual_features=dofa_s1,
)

torch.cuda.empty_cache()


# ============================================================
# TRAIN S2 MCQ
# ============================================================

s2_mcq_result = train_mcq_specialist(
    modality_name="S2",
    visual_features=dofa_s2,
)

torch.cuda.empty_cache()


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("S1/S2 OPTION-AWARE MCQ TRAINING FINISHED")
print("=" * 70)

print(
    "S1 best epoch:",
    s1_mcq_result["best_epoch"]
)

print(
    "S1 best val accuracy:",
    f"{s1_mcq_result['best_val_accuracy']:.4f}"
)

print(
    "S2 best epoch:",
    s2_mcq_result["best_epoch"]
)

print(
    "S2 best val accuracy:",
    f"{s2_mcq_result['best_val_accuracy']:.4f}"
)

print("\nSaved under:")
print(TRAIN_ROOT)

print("\nNOTE:")
print(
    "Both MCQ specialists used the four actual cached "
    "option embeddings A/B/C/D and completed all 10 epochs."
)