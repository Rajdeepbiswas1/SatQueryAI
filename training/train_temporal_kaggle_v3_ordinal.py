from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/kaggle/working")
DATA_ROOT = ROOT / "data"

TRAIN_MANIFEST = DATA_ROOT / "processed" / "temporal_split" / "train_manifest.csv"
VAL_MANIFEST = DATA_ROOT / "processed" / "temporal_split" / "val_manifest.csv"
FEATURE_ROOT = DATA_ROOT / "processed" / "temporal_features"
TRAIN_Q_ROOT = DATA_ROOT / "processed" / "temporal_question_embeddings"
VAL_Q_ROOT = DATA_ROOT / "processed" / "temporal_val_question_embeddings"

CHECKPOINT_ROOT = (
    ROOT / "models" / "checkpoints" / "temporal_change_vqa_v3_ordinal"
)
OUTPUT_ROOT = ROOT / "outputs" / "temporal_change_vqa_v3_ordinal"

CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
USE_AMP = True

MODEL_NAME = "TemporalChangeVQAv3OrdinalLoss"
ATTENTION_TEMPERATURE = 48.0

# V3 controlled loss change only.
# Adds a small ordinal-distance penalty to the two ordered ratio heads.
ORDINAL_LAMBDA = 0.10

BINARY = ("no", "yes")

LAND_COVER = (
    "buildings",
    "trees",
    "low_vegetation",
    "NVG_surface",
    "water",
    "playgrounds",
)

RATIO = (
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
    "80_to_90",
    "90_to_100",
)

RATIO_TYPE = (
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
)

QTYPE_TO_HEAD = {
    "change_or_not": "binary",
    "increase_or_not": "binary",
    "decrease_or_not": "binary",
    "smallest_change": "land_cover",
    "largest_change": "land_cover",
    "change_to_what": "land_cover",
    "change_ratio": "change_ratio",
    "change_ratio_types": "change_ratio_types",
}

VOCAB_TO_INDEX = {
    "binary": {v: i for i, v in enumerate(BINARY)},
    "land_cover": {v: i for i, v in enumerate(LAND_COVER)},
    "change_ratio": {v: i for i, v in enumerate(RATIO)},
    "change_ratio_types": {v: i for i, v in enumerate(RATIO_TYPE)},
}

HEAD_TO_QTYPES = defaultdict(list)
for qtype, head in QTYPE_TO_HEAD.items():
    HEAD_TO_QTYPES[head].append(qtype)

EXPECTED_PAIRS = 2968
EXPECTED_TRAIN_PAIRS = 1600
EXPECTED_VAL_PAIRS = 400
EXPECTED_TEST_PAIRS = 968


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required path not found ({label}): {path}"
        )


def normalize_pair_id(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    text = str(value).strip()

    if text.isdigit():
        return text.zfill(5)

    return text


def read_manifest(path: Path, label: str) -> pd.DataFrame:
    require_path(path, label)

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(f"{label} manifest is empty: {path}")

    if "question_id" not in df.columns:
        for candidate in (
            "question_index",
            "question_embedding_index",
            "qid",
        ):
            if candidate in df.columns:
                df["question_id"] = df[candidate]
                break

    if "pair_id" not in df.columns:
        for candidate in (
            "pair",
            "pairID",
            "image_pair_id",
            "image_id",
        ):
            if candidate in df.columns:
                df["pair_id"] = df[candidate]
                break

    if "qtype" not in df.columns:
        for candidate in ("question_type", "type"):
            if candidate in df.columns:
                df["qtype"] = df[candidate]
                break

    if "answer" not in df.columns:
        for candidate in ("label", "target"):
            if candidate in df.columns:
                df["answer"] = df[candidate]
                break

    required = {"pair_id", "question_id", "qtype", "answer"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"{label} manifest is missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["pair_id"] = df["pair_id"].map(normalize_pair_id)
    df["question_id"] = df["question_id"].astype(int)
    df["qtype"] = df["qtype"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()

    unknown_qtypes = sorted(
        set(df["qtype"]) - set(QTYPE_TO_HEAD)
    )

    if unknown_qtypes:
        raise RuntimeError(
            f"{label} manifest contains unknown question types: "
            f"{unknown_qtypes}"
        )

    return df


class TemporalFeatureStore:
    """
    Loads verified temporal DOFA feature shards into CPU RAM once.
    """

    def __init__(self, root: Path) -> None:
        require_path(root, "temporal feature root")

        files = sorted(root.glob("shard_*.pt"))
        if not files:
            files = sorted(root.glob("*.pt"))

        if not files:
            raise FileNotFoundError(
                f"No temporal feature shards found under {root}"
            )

        self.pair_ids: list[str] = []
        t1_parts: list[torch.Tensor] = []
        t2_parts: list[torch.Tensor] = []
        diff_parts: list[torch.Tensor] = []

        print()
        print("=" * 74)
        print("LOADING TEMPORAL FEATURE SHARDS")
        print("=" * 74)

        for index, path in enumerate(files, start=1):
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            pair_ids = data.get("pair_ids")
            t1 = data.get("t1")

            if t1 is None:
                t1 = data.get("t1_features")

            t2 = data.get("t2")

            if t2 is None:
                t2 = data.get("t2_features")

            diff = data.get("diff")

            if diff is None:
                diff = data.get("diff_features")

            if pair_ids is None or t1 is None or t2 is None or diff is None:
                raise RuntimeError(
                    f"{path}: expected pair_ids plus temporal feature tensors "
                    f"(t1/t2/diff or t1_features/t2_features/diff_features). "
                    f"Found: {sorted(data.keys())}"
                )

            if not (t1.shape == t2.shape == diff.shape):
                raise RuntimeError(
                    f"{path}: t1/t2/diff shape mismatch: "
                    f"{tuple(t1.shape)}, {tuple(t2.shape)}, "
                    f"{tuple(diff.shape)}"
                )

            if t1.ndim != 3 or tuple(t1.shape[1:]) != (196, 768):
                raise RuntimeError(
                    f"{path}: invalid feature shape {tuple(t1.shape)}; "
                    f"expected [N,196,768]"
                )

            if len(pair_ids) != t1.shape[0]:
                raise RuntimeError(
                    f"{path}: pair ID count {len(pair_ids)} "
                    f"does not match tensor rows {t1.shape[0]}"
                )

            self.pair_ids.extend(
                normalize_pair_id(x)
                for x in pair_ids
            )
            t1_parts.append(t1.contiguous())
            t2_parts.append(t2.contiguous())
            diff_parts.append(diff.contiguous())

            print(
                f"Shard {index:02d}/{len(files):02d} | "
                f"pairs={t1.shape[0]:4d}"
            )

        self.t1 = torch.cat(t1_parts, dim=0)
        self.t2 = torch.cat(t2_parts, dim=0)
        self.diff = torch.cat(diff_parts, dim=0)

        if len(self.pair_ids) != EXPECTED_PAIRS:
            raise RuntimeError(
                f"Expected {EXPECTED_PAIRS} feature pairs, "
                f"found {len(self.pair_ids)}"
            )

        if len(set(self.pair_ids)) != len(self.pair_ids):
            raise RuntimeError("Duplicate temporal pair IDs detected.")

        self.lookup = {
            pair_id: index
            for index, pair_id in enumerate(self.pair_ids)
        }

        print()
        print(f"Feature pairs: {len(self.pair_ids)}")
        print(f"T1 shape: {tuple(self.t1.shape)}")
        print(f"T2 shape: {tuple(self.t2.shape)}")
        print(f"Diff shape: {tuple(self.diff.shape)}")
        print(f"Feature dtype: {self.t1.dtype}")

        ram_gib = (
            (
                self.t1.numel()
                + self.t2.numel()
                + self.diff.numel()
            )
            * self.t1.element_size()
            / (1024 ** 3)
        )

        print(f"Feature RAM footprint: {ram_gib:.2f} GiB")


class QuestionEmbeddingStore:
    """
    Loads one verified question cache into CPU RAM.
    """

    def __init__(
        self,
        root: Path,
        expected_rows: int,
        label: str,
    ) -> None:
        require_path(root, label)

        files = sorted(root.glob("shard_*.pt"))
        if not files:
            files = sorted(root.glob("*.pt"))

        if not files:
            raise FileNotFoundError(
                f"No question embedding shards found under {root}"
            )

        ids: list[int] = []
        embedding_parts: list[torch.Tensor] = []

        print()
        print("=" * 74)
        print(label.upper())
        print("=" * 74)

        for index, path in enumerate(files, start=1):
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            question_ids = data.get("question_ids")
            if question_ids is None:
                question_ids = data.get("ids")

            question_embeddings = data.get("question_embeddings")
            if question_embeddings is None:
                question_embeddings = data.get("embeddings")

            if question_ids is None or question_embeddings is None:
                raise RuntimeError(
                    f"{path}: expected question IDs and embeddings using "
                    f"question_ids/question_embeddings or ids/embeddings. "
                    f"Found: {sorted(data.keys())}"
                )

            if question_embeddings.ndim != 2:
                raise RuntimeError(
                    f"{path}: invalid embedding shape "
                    f"{tuple(question_embeddings.shape)}"
                )

            if question_embeddings.shape[1] != 768:
                raise RuntimeError(
                    f"{path}: expected embedding width 768, "
                    f"found {question_embeddings.shape[1]}"
                )

            if len(question_ids) != question_embeddings.shape[0]:
                raise RuntimeError(
                    f"{path}: question ID / embedding row mismatch"
                )

            ids.extend(int(x) for x in question_ids)
            embedding_parts.append(
                question_embeddings.float().contiguous()
            )

            print(
                f"Shard {index:02d}/{len(files):02d} | "
                f"rows={question_embeddings.shape[0]:5d}"
            )

        if len(ids) != expected_rows:
            raise RuntimeError(
                f"{label}: expected {expected_rows} rows, "
                f"found {len(ids)}"
            )

        if len(set(ids)) != len(ids):
            raise RuntimeError(
                f"{label}: duplicate question IDs detected."
            )

        order = np.argsort(np.asarray(ids))
        stacked = torch.cat(embedding_parts, dim=0)

        self.ids = [ids[i] for i in order]
        self.embeddings = stacked[
            torch.as_tensor(order, dtype=torch.long)
        ].contiguous()

        if self.ids != list(range(expected_rows)):
            raise RuntimeError(
                f"{label}: question IDs are not continuous "
                f"from 0 to {expected_rows - 1}"
            )

        print(f"Question IDs: {len(self.ids)}")
        print(f"Embedding matrix: {tuple(self.embeddings.shape)}")
        print(f"Embedding dtype: {self.embeddings.dtype}")

    def get(self, question_id: int) -> torch.Tensor:
        if not 0 <= question_id < len(self.ids):
            raise KeyError(
                f"Question ID out of range: {question_id}"
            )

        return self.embeddings[question_id]


class TemporalRAMDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_store: TemporalFeatureStore,
        question_store: QuestionEmbeddingStore,
    ) -> None:
        self.rows = manifest.reset_index(drop=True)
        self.feature_store = feature_store
        self.question_store = question_store

        self.feature_indices: list[int] = []

        for pair_id in self.rows["pair_id"]:
            if pair_id not in self.feature_store.lookup:
                raise RuntimeError(
                    f"Missing feature pair for manifest pair_id={pair_id}"
                )

            self.feature_indices.append(
                self.feature_store.lookup[pair_id]
            )

        max_qid = len(self.question_store.ids) - 1
        qids = self.rows["question_id"].to_numpy()

        if np.any(qids < 0) or np.any(qids > max_qid):
            raise RuntimeError(
                "Manifest contains question IDs outside the cache range."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        feature_index = self.feature_indices[index]

        return {
            "pair_id": row["pair_id"],
            "question_id": int(row["question_id"]),
            "qtype": row["qtype"],
            "answer": row["answer"],
            "t1": self.feature_store.t1[feature_index],
            "t2": self.feature_store.t2[feature_index],
            "diff": self.feature_store.diff[feature_index],
            "question_embedding": self.question_store.get(
                int(row["question_id"])
            ),
        }


def collate_batch(
    batch: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "t1": torch.stack(
            [item["t1"] for item in batch],
            dim=0,
        ),
        "t2": torch.stack(
            [item["t2"] for item in batch],
            dim=0,
        ),
        "diff": torch.stack(
            [item["diff"] for item in batch],
            dim=0,
        ),
        "question_embedding": torch.stack(
            [item["question_embedding"] for item in batch],
            dim=0,
        ),
        "qtype": [item["qtype"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "pair_id": [item["pair_id"] for item in batch],
        "question_id": [
            item["question_id"]
            for item in batch
        ],
    }


def encode_labels(
    qtypes: list[str],
    answers: list[str],
) -> dict[str, torch.Tensor]:
    batch_size = len(qtypes)

    targets = {
        head: torch.full(
            (batch_size,),
            -100,
            dtype=torch.long,
        )
        for head in VOCAB_TO_INDEX
    }

    for index, (qtype, answer) in enumerate(
        zip(qtypes, answers)
    ):
        head = QTYPE_TO_HEAD[qtype]
        vocabulary = VOCAB_TO_INDEX[head]

        if answer not in vocabulary:
            raise RuntimeError(
                f"Unexpected answer {answer!r} for "
                f"question type {qtype!r}; "
                f"valid values={list(vocabulary)}"
            )

        targets[head][index] = vocabulary[answer]

    return targets


def build_class_weights(
    manifest: pd.DataFrame,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}

    for head, vocabulary in (
        ("binary", BINARY),
        ("land_cover", LAND_COVER),
        ("change_ratio", RATIO),
        ("change_ratio_types", RATIO_TYPE),
    ):
        subset = manifest[
            manifest["qtype"].isin(
                HEAD_TO_QTYPES[head]
            )
        ]

        counts = Counter(
            subset["answer"].tolist()
        )

        frequencies = np.array(
            [
                max(counts.get(label, 0), 1)
                for label in vocabulary
            ],
            dtype=np.float64,
        )

        raw = 1.0 / np.sqrt(frequencies)
        raw /= raw.mean()

        weights[head] = torch.tensor(
            raw,
            dtype=torch.float32,
        )

    return weights


def ordinal_distance_penalty(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable ordinal penalty.

    The classes in the ratio heads are ordered bins. The penalty
    compares the probability-weighted expected class index with
    the target index, normalized to [0, 1].

    This does not change the model architecture.
    """
    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    positions = torch.arange(
        logits.shape[-1],
        device=logits.device,
        dtype=probabilities.dtype,
    )

    expected_position = (
        probabilities * positions
    ).sum(dim=-1)

    target_position = target.to(
        probabilities.dtype
    )

    scale = float(
        max(logits.shape[-1] - 1, 1)
    )

    normalized_error = (
        expected_position
        - target_position
    ) / scale

    return (
        normalized_error.square().mean()
    )


def compute_masked_balanced_loss(
    answer_logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, int]:
    """
    V2 balanced CE baseline plus a small ordinal penalty
    for the ordered change-ratio heads only.
    """
    losses: list[torch.Tensor] = []
    active_count = 0

    for head, logits in answer_logits.items():
        target = targets[head]
        mask = target != -100

        count = int(mask.sum().item())
        if count == 0:
            continue

        active_count += count

        masked_logits = logits[mask]
        masked_target = target[mask]

        head_loss = nn.functional.cross_entropy(
            masked_logits,
            masked_target,
            weight=class_weights[head].to(
                logits.device
            ),
        )

        if head in (
            "change_ratio",
            "change_ratio_types",
        ):
            head_loss = (
                head_loss
                + ORDINAL_LAMBDA
                * ordinal_distance_penalty(
                    masked_logits,
                    masked_target,
                )
            )

        losses.append(head_loss)

    if not losses:
        raise RuntimeError("No active loss head in batch.")

    return torch.stack(losses).mean(), active_count


def empty_metric_state() -> dict[str, Any]:
    return {
        head: {
            "correct": 0,
            "total": 0,
        }
        for head in VOCAB_TO_INDEX
    }


def update_metric_state(
    state: dict[str, Any],
    answer_logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for head, logits in answer_logits.items():
            target = targets[head]
            mask = target != -100

            if not bool(mask.any()):
                continue

            prediction = logits.argmax(dim=-1)

            target_cpu = target[mask].detach().cpu()
            pred_cpu = prediction[mask].detach().cpu()

            state[head]["correct"] += int(
                (target_cpu == pred_cpu).sum().item()
            )
            state[head]["total"] += int(
                target_cpu.numel()
            )


def empty_confusion_state() -> dict[str, np.ndarray]:
    return {
        head: np.zeros(
            (
                len(VOCAB_TO_INDEX[head]),
                len(VOCAB_TO_INDEX[head]),
            ),
            dtype=np.int64,
        )
        for head in VOCAB_TO_INDEX
    }


def update_confusion_state(
    confusion: dict[str, np.ndarray],
    answer_logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for head, logits in answer_logits.items():
            target = targets[head]
            mask = target != -100

            if not bool(mask.any()):
                continue

            pred = logits.argmax(dim=-1)

            t = target[mask].detach().cpu().tolist()
            p = pred[mask].detach().cpu().tolist()

            matrix = confusion[head]

            for true_label, predicted_label in zip(t, p):
                matrix[
                    int(true_label),
                    int(predicted_label),
                ] += 1


def summarize_metrics(
    state: dict[str, Any],
    confusion: dict[str, np.ndarray],
) -> dict[str, Any]:
    head_metrics: dict[str, Any] = {}

    total_correct = 0
    total_count = 0

    for head in VOCAB_TO_INDEX:
        correct = state[head]["correct"]
        total = state[head]["total"]

        total_correct += correct
        total_count += total

        matrix = confusion[head]
        recalls: list[float] = []

        for class_id in range(matrix.shape[0]):
            support = int(matrix[class_id].sum())

            if support == 0:
                continue

            recalls.append(
                float(
                    matrix[class_id, class_id]
                    / support
                )
            )

        balanced = (
            float(np.mean(recalls))
            if recalls
            else float("nan")
        )

        head_metrics[head] = {
            "accuracy": (
                correct / total
                if total
                else float("nan")
            ),
            "balanced_accuracy": balanced,
            "samples": total,
        }

    valid_balanced = [
        value["balanced_accuracy"]
        for value in head_metrics.values()
        if not math.isnan(
            value["balanced_accuracy"]
        )
    ]

    return {
        "accuracy": (
            total_correct / total_count
            if total_count
            else float("nan")
        ),
        "balanced_accuracy": (
            float(np.mean(valid_balanced))
            if valid_balanced
            else float("nan")
        ),
        "head_metrics": head_metrics,
    }


def ensure_feature_fusion() -> None:
    """
    Recreate the exact verified TemporalFusion dependency only when
    it is absent from the Kaggle working tree.
    """

    target = (
        ROOT
        / "src"
        / "models"
        / "feature_fusion.py"
    )

    if target.exists():
        return

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = """from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.s1_proj = nn.Linear(input_dim, hidden_dim)
        self.s2_proj = nn.Linear(input_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
    ) -> torch.Tensor:
        if s1.ndim != 3 or s2.ndim != 3:
            raise ValueError(
                "Expected [B,196,768] inputs."
            )

        if s1.shape != s2.shape:
            raise ValueError(
                f"Input shapes must match: "
                f"{tuple(s1.shape)} vs {tuple(s2.shape)}"
            )

        s1 = self.s1_proj(s1.float())
        s2 = self.s2_proj(s2.float())

        x = torch.cat(
            [s1, s2],
            dim=-1,
        )

        return self.fusion(x)


class TemporalFusion(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.t1_proj = nn.Linear(
            input_dim,
            hidden_dim,
        )
        self.t2_proj = nn.Linear(
            input_dim,
            hidden_dim,
        )
        self.diff_proj = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_dim * 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        diff: torch.Tensor,
    ) -> torch.Tensor:
        if (
            t1.ndim != 3
            or t2.ndim != 3
            or diff.ndim != 3
        ):
            raise ValueError(
                "Expected all inputs to have shape [B,196,768]."
            )

        if not (
            t1.shape == t2.shape == diff.shape
        ):
            raise ValueError(
                f"Temporal input shapes must match: "
                f"T1={tuple(t1.shape)}, "
                f"T2={tuple(t2.shape)}, "
                f"Diff={tuple(diff.shape)}"
            )

        t1 = self.t1_proj(t1.float())
        t2 = self.t2_proj(t2.float())
        diff = self.diff_proj(diff.float())

        x = torch.cat(
            [t1, t2, diff],
            dim=-1,
        )

        return self.fusion(x)
"""

    target.write_text(
        source,
        encoding="utf-8",
    )


def import_model() -> tuple[type[nn.Module], Any]:
    src_root = ROOT / "src"

    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    try:
        from src.models.temporal_change_vqa_v2 import (
            TemporalChangeVQAv2,
            count_trainable_parameters,
        )
    except ImportError as exc:
        raise ImportError(
            "Could not import TemporalChangeVQAv2. "
            "Upload temporal_change_vqa_v2.py to "
            "/kaggle/working/src/models/."
        ) from exc

    return TemporalChangeVQAv2, count_trainable_parameters


def run_initial_forward_check(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    print()
    print("=" * 74)
    print("INITIAL REAL-DATA FORWARD CHECK")
    print("=" * 74)

    batch = next(iter(loader))

    t1 = batch["t1"][:4].to(device)
    t2 = batch["t2"][:4].to(device)
    diff = batch["diff"][:4].to(device)
    question = batch["question_embedding"][:4].to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(
            t1=t1,
            t2=t2,
            diff=diff,
            question=question,
        )

    expected = {
        "query_attention": (4, 14, 14),
        "change_logits": (4, 14, 14),
        "fused_tokens": (4, 196, 256),
        "joint": (4, 256),
        "answer_logits[binary]": (4, 2),
        "answer_logits[change_ratio]": (4, 11),
        "answer_logits[change_ratio_types]": (4, 9),
        "answer_logits[land_cover]": (4, 6),
    }

    observed = {
        "query_attention": tuple(
            outputs["query_attention"].shape
        ),
        "change_logits": tuple(
            outputs["change_logits"].shape
        ),
        "fused_tokens": tuple(
            outputs["fused_tokens"].shape
        ),
        "joint": tuple(
            outputs["joint"].shape
        ),
        "answer_logits[binary]": tuple(
            outputs["answer_logits"]["binary"].shape
        ),
        "answer_logits[change_ratio]": tuple(
            outputs["answer_logits"]["change_ratio"].shape
        ),
        "answer_logits[change_ratio_types]": tuple(
            outputs["answer_logits"]["change_ratio_types"].shape
        ),
        "answer_logits[land_cover]": tuple(
            outputs["answer_logits"]["land_cover"].shape
        ),
    }

    for name, expected_shape in expected.items():
        actual_shape = observed[name]

        print(
            f"{name:34s}: {actual_shape}"
        )

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Interface mismatch for {name}: "
                f"expected {expected_shape}, "
                f"got {actual_shape}"
            )

    for name, tensor in outputs["answer_logits"].items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Non-finite answer logits in head {name}"
            )

    if not torch.isfinite(
        outputs["query_attention"]
    ).all():
        raise RuntimeError("Non-finite query attention.")

    if not torch.isfinite(
        outputs["change_logits"]
    ).all():
        raise RuntimeError(
            "Non-finite change evidence logits."
        )

    print("Real-data forward check: PASS")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    train: bool,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    model.train(train)

    total_loss = 0.0
    total_active = 0

    metric_state = empty_metric_state()
    confusion = empty_confusion_state()

    for batch in loader:
        t1 = batch["t1"].to(
            device,
            non_blocking=True,
        )
        t2 = batch["t2"].to(
            device,
            non_blocking=True,
        )
        diff = batch["diff"].to(
            device,
            non_blocking=True,
        )
        question = batch["question_embedding"].to(
            device,
            non_blocking=True,
        )

        targets = encode_labels(
            batch["qtype"],
            batch["answer"],
        )

        targets = {
            head: target.to(
                device,
                non_blocking=True,
            )
            for head, target in targets.items()
        }

        if train:
            optimizer.zero_grad(
                set_to_none=True
            )

        amp_enabled = (
            USE_AMP
            and device.type == "cuda"
        )

        with torch.autocast(
            device_type=device.type,
            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16
            ),
            enabled=amp_enabled,
        ):
            outputs = model(
                t1=t1,
                t2=t2,
                diff=diff,
                question=question,
            )

            loss, active_count = (
                compute_masked_balanced_loss(
                    outputs["answer_logits"],
                    targets,
                    class_weights,
                )
            )

        if train:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += (
            float(loss.detach().item())
            * active_count
        )
        total_active += active_count

        update_metric_state(
            metric_state,
            outputs["answer_logits"],
            targets,
        )

        update_confusion_state(
            confusion,
            outputs["answer_logits"],
            targets,
        )

    metrics = summarize_metrics(
        metric_state,
        confusion,
    )

    mean_loss = (
        total_loss / total_active
        if total_active
        else float("nan")
    )

    return mean_loss, metrics, confusion


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_metrics: dict[str, Any],
    parameter_count: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics[
                "balanced_accuracy"
            ],
            "parameter_count": parameter_count,
            "model_name": MODEL_NAME,
            "attention_temperature": (
                ATTENTION_TEMPERATURE
            ),
            "seed": SEED,
            "config": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "loss": (
                    "moderated inverse-sqrt "
                    "class-balanced cross-entropy "
                    "+ ordinal ratio penalty"
                ),
                "ordinal_lambda": ORDINAL_LAMBDA,
                "architecture": (
                    "v2: direct normalized "
                    "question_projection query"
                ),
            },
        },
        path,
    )


def main() -> None:
    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 74)
    print("TEMPORAL CHANGE VQA v3 — ORDINAL-REGULARIZED CLASS-BALANCED TRAINING")
    print("=" * 74)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"CUDA device: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Patience: {PATIENCE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")

    print(
        "Loss: moderated class-balanced "
        "cross-entropy using inverse-sqrt frequency weights "
        "+ ordinal penalty on ratio heads"
    )
    print(
        f"Ordinal lambda: {ORDINAL_LAMBDA}"
    )

    print(
        "Architecture change: remove attention_query; "
        "use normalized question_projection directly; "
        "temperature=48"
    )

    ensure_feature_fusion()
    TemporalChangeVQAv2, count_params = import_model()

    for path, label in (
        (TRAIN_MANIFEST, "train_manifest"),
        (VAL_MANIFEST, "val_manifest"),
        (FEATURE_ROOT, "temporal_features"),
        (TRAIN_Q_ROOT, "training_question_embeddings"),
        (VAL_Q_ROOT, "validation_question_embeddings"),
    ):
        require_path(path, label)

    train_manifest = read_manifest(
        TRAIN_MANIFEST,
        "train",
    )
    val_manifest = read_manifest(
        VAL_MANIFEST,
        "validation",
    )

    train_pairs = set(
        train_manifest["pair_id"]
    )
    val_pairs = set(
        val_manifest["pair_id"]
    )

    print()
    print("=" * 74)
    print("VERIFYING TRAIN / VALIDATION SPLIT")
    print("=" * 74)
    print(
        f"Training rows: {len(train_manifest):,}"
    )
    print(
        f"Validation rows: {len(val_manifest):,}"
    )
    print(
        f"Training pairs: {len(train_pairs):,}"
    )
    print(
        f"Validation pairs: {len(val_pairs):,}"
    )

    overlap = train_pairs & val_pairs

    if overlap:
        raise RuntimeError(
            f"Train/validation pair overlap detected: "
            f"{sorted(overlap)[:10]}"
        )

    if len(train_pairs) != EXPECTED_TRAIN_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_PAIRS} train pairs, "
            f"found {len(train_pairs)}"
        )

    if len(val_pairs) != EXPECTED_VAL_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL_PAIRS} validation pairs, "
            f"found {len(val_pairs)}"
        )

    print("Train/validation pair split: PASS")

    feature_store = TemporalFeatureStore(
        FEATURE_ROOT
    )

    all_required_pairs = (
        train_pairs | val_pairs
    )

    missing_features = (
        all_required_pairs
        - set(feature_store.lookup)
    )

    if missing_features:
        raise RuntimeError(
            f"Missing feature pairs: "
            f"{sorted(missing_features)[:10]}"
        )

    print("Manifest-to-feature mapping: PASS")

    train_question_store = QuestionEmbeddingStore(
        TRAIN_Q_ROOT,
        expected_rows=len(train_manifest),
        label="training question embeddings",
    )

    val_question_store = QuestionEmbeddingStore(
        VAL_Q_ROOT,
        expected_rows=len(val_manifest),
        label="validation question embeddings",
    )

    train_dataset = TemporalRAMDataset(
        train_manifest,
        feature_store,
        train_question_store,
    )

    val_dataset = TemporalRAMDataset(
        val_manifest,
        feature_store,
        val_question_store,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    print(
        f"Train batches: {len(train_loader):,}"
    )
    print(
        f"Validation batches: {len(val_loader):,}"
    )

    class_weights = build_class_weights(
        train_manifest
    )

    class_weight_json = {
        head: [float(x) for x in weights.tolist()]
        for head, weights in class_weights.items()
    }

    print()
    print("=" * 74)
    print("BALANCED CLASS WEIGHTS")
    print("=" * 74)

    for head, values in class_weight_json.items():
        print(f"{head}: {values}")

    (
        OUTPUT_ROOT / "class_weights.json"
    ).write_text(
        json.dumps(
            class_weight_json,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 74)
    print("BUILDING TEMPORAL CHANGE VQA v2")
    print("=" * 74)

    model = TemporalChangeVQAv2(
        input_dim=768,
        hidden_dim=256,
        num_spatial_tokens=196,
        grid_size=14,
    ).to(device)

    parameter_count = count_params(model)

    print(
        f"Attention temperature: "
        f"{model.ATTENTION_TEMPERATURE}"
    )
    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    if parameter_count != 1_679_133:
        raise RuntimeError(
            "Architecture parameter count mismatch. "
            f"Expected 1,679,133, got {parameter_count}."
        )

    if hasattr(model, "attention_query"):
        raise RuntimeError(
            "v3 unexpectedly contains attention_query."
        )

    if parameter_count != 1_679_133:
        raise RuntimeError(
            "V3 parameter count drift detected."
        )

    print("v3 architecture parameter count: PASS")
    print(
        f"V3 loss-only experiment; ordinal lambda={ORDINAL_LAMBDA}"
    )

    run_initial_forward_check(
        model,
        train_loader,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if (
            USE_AMP
            and device.type == "cuda"
        )
        else None
    )

    history: list[dict[str, Any]] = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_path = (
        CHECKPOINT_ROOT / "best.pt"
    )
    latest_path = (
        CHECKPOINT_ROOT / "latest.pt"
    )

    print()
    print("=" * 74)
    print("STARTING TRAINING")
    print("=" * 74)

    training_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start = time.time()

        train_loss, train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            class_weights=class_weights,
            optimizer=optimizer,
            scaler=scaler,
            train=True,
        )

        val_start = time.time()

        with torch.no_grad():
            val_loss, val_metrics, _ = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                class_weights=class_weights,
                optimizer=None,
                scaler=None,
                train=False,
            )

        val_time = time.time() - val_start
        epoch_time = time.time() - epoch_start

        print()
        print("-" * 74)
        print(f"Epoch {epoch:02d}/{EPOCHS}")
        print(
            f"Train loss:              "
            f"{train_loss:.6f}"
        )
        print(
            f"Train accuracy:          "
            f"{train_metrics['accuracy']:.6f}"
        )
        print(
            f"Train balanced accuracy: "
            f"{train_metrics['balanced_accuracy']:.6f}"
        )
        print(
            f"Validation loss:          "
            f"{val_loss:.6f}"
        )
        print(
            f"Validation accuracy:      "
            f"{val_metrics['accuracy']:.6f}"
        )
        print(
            f"Validation balanced acc:  "
            f"{val_metrics['balanced_accuracy']:.6f}"
        )
        print(
            f"Epoch time:               "
            f"{epoch_time:.2f}s"
        )
        print(
            f"Validation time:          "
            f"{val_time:.2f}s"
        )

        print()
        print("Validation metrics by head:")

        for head, values in (
            val_metrics["head_metrics"].items()
        ):
            print(
                f"  {head:22s} "
                f"accuracy={values['accuracy']:.4f} "
                f"balanced={values['balanced_accuracy']:.4f} "
                f"samples={values['samples']}"
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics[
                "accuracy"
            ],
            "train_balanced_accuracy": train_metrics[
                "balanced_accuracy"
            ],
            "val_loss": val_loss,
            "val_accuracy": val_metrics[
                "accuracy"
            ],
            "val_balanced_accuracy": val_metrics[
                "balanced_accuracy"
            ],
            "epoch_time_seconds": epoch_time,
            "validation_time_seconds": val_time,
        }

        for head, values in (
            val_metrics["head_metrics"].items()
        ):
            record[
                f"val_{head}_accuracy"
            ] = values["accuracy"]

            record[
                f"val_{head}_balanced_accuracy"
            ] = values["balanced_accuracy"]

        history.append(record)

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            val_loss,
            val_metrics,
            parameter_count,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                val_loss,
                val_metrics,
                parameter_count,
            )

            print()
            print("NEW BEST CHECKPOINT")
            print(
                f"Best validation loss: "
                f"{best_val_loss:.6f}"
            )
            print(
                f"Checkpoint: {best_path}"
            )

        else:
            patience_counter += 1

            print()
            print(
                "No validation-loss improvement."
            )
            print(
                f"Patience: "
                f"{patience_counter}/{PATIENCE}"
            )

            if patience_counter >= PATIENCE:
                print()
                print(
                    "Early stopping triggered."
                )
                break

    total_training_time = (
        time.time() - training_start
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        OUTPUT_ROOT / "training_history.csv",
        index=False,
    )

    summary = {
        "model_name": MODEL_NAME,
        "attention_temperature": ATTENTION_TEMPERATURE,
        "parameter_count": parameter_count,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "total_training_time_seconds": (
            total_training_time
        ),
        "train_rows": len(train_manifest),
        "validation_rows": len(val_manifest),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "reserved_test_pairs": EXPECTED_TEST_PAIRS,
        "feature_pairs": len(feature_store.pair_ids),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": (
            "moderated inverse-sqrt "
            "class-balanced cross-entropy "
            "+ ordinal ratio penalty"
        ),
        "ordinal_lambda": ORDINAL_LAMBDA,
        "architecture_change": (
            "none relative to V2; "
            "V2 architecture retained unchanged"
        ),
        "checkpoint": str(best_path),
    }

    (
        OUTPUT_ROOT / "training_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 74)
    print("TEMPORAL CHANGE VQA v3 TRAINING COMPLETE")
    print("=" * 74)
    print(
        f"Epochs completed: {len(history)}"
    )
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )
    print(
        f"Total training time: "
        f"{total_training_time:.2f}s"
    )
    print()
    print(
        f"Best checkpoint: {best_path}"
    )
    print(
        f"Training history: "
        f"{OUTPUT_ROOT / 'training_history.csv'}"
    )
    print(
        f"Summary: "
        f"{OUTPUT_ROOT / 'training_summary.json'}"
    )
    print()
    print("DONE")


if __name__ == "__main__":
    main()
