from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")

SRC_DIR = ROOT / "src"

TRAIN_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "temporal_split"
    / "train_manifest.csv"
)

VAL_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "temporal_split"
    / "val_manifest.csv"
)

FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "temporal_features"
)

TRAIN_QUESTION_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "temporal_question_embeddings"
)

VAL_QUESTION_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "temporal_val_question_embeddings"
)

OUTPUT_DIR = (
    ROOT
    / "models"
    / "checkpoints"
    / "temporal_change_vqa"
)

HISTORY_FILE = (
    ROOT
    / "outputs"
    / "temporal_training_history.csv"
)

SUMMARY_FILE = (
    ROOT
    / "outputs"
    / "temporal_training_summary.json"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 0

EPOCHS = 20
PATIENCE = 4

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PIN_MEMORY = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# EXACT VERIFIED LABEL VOCABULARIES
# ============================================================

BINARY_CLASSES = (
    "no",
    "yes",
)

LAND_COVER_CLASSES = (
    "buildings",
    "trees",
    "low_vegetation",
    "NVG_surface",
    "water",
    "playgrounds",
)

RATIO_CLASSES = (
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

RATIO_TYPE_CLASSES = (
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


QUESTION_TYPE_TO_HEAD = {
    "change_or_not": "binary",
    "increase_or_not": "binary",
    "decrease_or_not": "binary",
    "smallest_change": "land_cover",
    "largest_change": "land_cover",
    "change_to_what": "land_cover",
    "change_ratio": "ratio",
    "change_ratio_types": "ratio_type",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducible CUDA behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL IMPORT
# ============================================================

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.temporal_change_vqa import TemporalChangeVQA


# ============================================================
# IN-MEMORY TEMPORAL FEATURE STORE
# ============================================================

class TemporalFeatureStore:
    """
    Loads ALL temporal feature shards once into CPU RAM.

    Verified shard schema:

        pair_ids
        t1_features
        t2_features
        diff_features

    Each feature tensor:

        [N, 196, 768]

    dtype:

        float16
    """

    def __init__(
        self,
        feature_dir: Path,
    ) -> None:

        self.feature_dir = Path(feature_dir)

        if not self.feature_dir.exists():
            raise FileNotFoundError(
                f"Feature directory not found:\n"
                f"{self.feature_dir}"
            )

        shard_paths = sorted(
            self.feature_dir.glob("shard_*.pt")
        )

        if not shard_paths:
            raise FileNotFoundError(
                f"No temporal feature shards found in:\n"
                f"{self.feature_dir}"
            )

        print()
        print("=" * 72)
        print("LOADING ALL TEMPORAL FEATURES INTO RAM")
        print("=" * 72)

        print("Feature directory:", self.feature_dir)
        print("Shard count:", len(shard_paths))

        pair_records = []

        total_pairs = 0

        for shard_index, shard_path in enumerate(
            shard_paths,
            start=1,
        ):

            load_start = time.perf_counter()

            shard = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )

            required = {
                "pair_ids",
                "t1_features",
                "t2_features",
                "diff_features",
            }

            missing = required.difference(
                shard.keys()
            )

            if missing:
                raise RuntimeError(
                    f"Missing keys in {shard_path.name}: "
                    f"{sorted(missing)}"
                )

            pair_ids = [
                int(x)
                for x in shard["pair_ids"]
            ]

            t1 = shard["t1_features"]
            t2 = shard["t2_features"]
            diff = shard["diff_features"]

            n = len(pair_ids)

            expected_shape = (
                n,
                196,
                768,
            )

            if tuple(t1.shape) != expected_shape:
                raise RuntimeError(
                    f"Invalid T1 shape in "
                    f"{shard_path.name}: "
                    f"{tuple(t1.shape)}"
                )

            if tuple(t2.shape) != expected_shape:
                raise RuntimeError(
                    f"Invalid T2 shape in "
                    f"{shard_path.name}: "
                    f"{tuple(t2.shape)}"
                )

            if tuple(diff.shape) != expected_shape:
                raise RuntimeError(
                    f"Invalid diff shape in "
                    f"{shard_path.name}: "
                    f"{tuple(diff.shape)}"
                )

            if t1.dtype != torch.float16:
                raise RuntimeError(
                    f"Expected float16 T1 features in "
                    f"{shard_path.name}, got {t1.dtype}"
                )

            if t2.dtype != torch.float16:
                raise RuntimeError(
                    f"Expected float16 T2 features in "
                    f"{shard_path.name}, got {t2.dtype}"
                )

            if diff.dtype != torch.float16:
                raise RuntimeError(
                    f"Expected float16 diff features in "
                    f"{shard_path.name}, got {diff.dtype}"
                )

            for row_index, pair_id in enumerate(pair_ids):

                pair_records.append(
                    (
                        pair_id,
                        t1[row_index],
                        t2[row_index],
                        diff[row_index],
                    )
                )

            total_pairs += n

            elapsed = (
                time.perf_counter()
                - load_start
            )

            print(
                f"Shard {shard_index:02d}/"
                f"{len(shard_paths):02d} "
                f"| pairs={n:4d} "
                f"| elapsed={elapsed:.2f}s"
            )

        if total_pairs != 2968:
            raise RuntimeError(
                f"Expected exactly 2968 temporal pairs, "
                f"found {total_pairs}"
            )

        # ----------------------------------------------------
        # Sort by actual numeric pair ID.
        # ----------------------------------------------------

        pair_records.sort(
            key=lambda x: x[0]
        )

        pair_ids = [
            x[0]
            for x in pair_records
        ]

        if len(set(pair_ids)) != len(pair_ids):
            raise RuntimeError(
                "Duplicate pair IDs detected."
            )

        # ----------------------------------------------------
        # Stack into contiguous CPU tensors.
        # ----------------------------------------------------

        self.pair_ids = pair_ids

        self.pair_id_to_index = {
            pair_id: index
            for index, pair_id in enumerate(
                pair_ids
            )
        }

        print()
        print("Stacking temporal features...")

        stack_start = time.perf_counter()

        self.t1 = torch.stack(
            [
                x[1]
                for x in pair_records
            ],
            dim=0,
        )

        self.t2 = torch.stack(
            [
                x[2]
                for x in pair_records
            ],
            dim=0,
        )

        self.diff = torch.stack(
            [
                x[3]
                for x in pair_records
            ],
            dim=0,
        )

        stack_elapsed = (
            time.perf_counter()
            - stack_start
        )

        expected_shape = (
            2968,
            196,
            768,
        )

        if tuple(self.t1.shape) != expected_shape:
            raise RuntimeError(
                f"Final T1 shape incorrect: "
                f"{tuple(self.t1.shape)}"
            )

        if tuple(self.t2.shape) != expected_shape:
            raise RuntimeError(
                f"Final T2 shape incorrect: "
                f"{tuple(self.t2.shape)}"
            )

        if tuple(self.diff.shape) != expected_shape:
            raise RuntimeError(
                f"Final diff shape incorrect: "
                f"{tuple(self.diff.shape)}"
            )

        print(
            "T1 shape:",
            tuple(self.t1.shape),
        )

        print(
            "T2 shape:",
            tuple(self.t2.shape),
        )

        print(
            "Diff shape:",
            tuple(self.diff.shape),
        )

        print(
            "Feature dtype:",
            self.t1.dtype,
        )

        print(
            "Feature stacking time: "
            f"{stack_elapsed:.2f}s"
        )

        # ----------------------------------------------------
        # RAM footprint.
        # ----------------------------------------------------

        feature_bytes = (
            self.t1.numel() * self.t1.element_size()
            + self.t2.numel() * self.t2.element_size()
            + self.diff.numel() * self.diff.element_size()
        )

        print(
            "Feature RAM footprint: "
            f"{feature_bytes / (1024 ** 3):.2f} GiB"
        )

    def lookup(
        self,
        pair_ids: torch.Tensor,
    ):
        """
        pair_ids:
            [B]

        Returns CPU float16 tensors:
            t1 [B,196,768]
            t2 [B,196,768]
            diff [B,196,768]
        """

        indices = [
            self.pair_id_to_index[
                int(pair_id)
            ]
            for pair_id in pair_ids.tolist()
        ]

        indices = torch.tensor(
            indices,
            dtype=torch.long,
        )

        return (
            self.t1[indices],
            self.t2[indices],
            self.diff[indices],
        )


# ============================================================
# QUESTION EMBEDDING STORE
# ============================================================

def load_question_embeddings(
    cache_dir: Path,
    expected_count: int,
    cache_name: str,
) -> torch.Tensor:

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"{cache_name} directory not found:\n"
            f"{cache_dir}"
        )

    shard_paths = sorted(
        cache_dir.glob("shard_*.pt")
    )

    if not shard_paths:
        raise FileNotFoundError(
            f"No question embedding shards found:\n"
            f"{cache_dir}"
        )

    print()
    print("=" * 72)
    print(f"LOADING {cache_name.upper()}")
    print("=" * 72)

    ids_parts = []
    embedding_parts = []

    for shard_path in shard_paths:

        payload = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        if {"ids", "question_embeddings"}.issubset(payload.keys()):
            ids = payload["ids"]
            embeddings = payload["question_embeddings"]

        elif {"question_ids", "embeddings"}.issubset(payload.keys()):
            ids = payload["question_ids"]
            embeddings = payload["embeddings"]

        else:
            raise RuntimeError(
                f"Unsupported question-cache schema in {shard_path.name}. "
                f"Found keys: {list(payload.keys())}"
            )
            
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings)

        if ids.ndim != 1:
            raise RuntimeError(
                f"Invalid IDs shape in "
                f"{shard_path.name}: "
                f"{tuple(ids.shape)}"
            )

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Invalid embedding rank in "
                f"{shard_path.name}: "
                f"{tuple(embeddings.shape)}"
            )

        if embeddings.shape[1] != 768:
            raise RuntimeError(
                f"Invalid embedding dimension in "
                f"{shard_path.name}: "
                f"{tuple(embeddings.shape)}"
            )

        if embeddings.shape[0] != ids.shape[0]:
            raise RuntimeError(
                f"ID/embedding mismatch in "
                f"{shard_path.name}"
            )

        if embeddings.dtype != torch.float32:
            raise RuntimeError(
                f"Expected float32 question "
                f"embeddings in {shard_path.name}"
            )

        ids_parts.append(ids)
        embedding_parts.append(
            embeddings
        )

    all_ids = torch.cat(
        ids_parts,
        dim=0,
    )

    all_embeddings = torch.cat(
        embedding_parts,
        dim=0,
    )

    if all_ids.numel() != expected_count:
        raise RuntimeError(
            f"{cache_name}: expected "
            f"{expected_count} IDs, found "
            f"{all_ids.numel()}"
        )

    expected_ids = torch.arange(
        expected_count,
        dtype=torch.long,
    )

    if not torch.equal(
        all_ids,
        expected_ids,
    ):
        raise RuntimeError(
            f"{cache_name}: IDs are not exactly "
            f"0..{expected_count - 1}"
        )

    if tuple(
        all_embeddings.shape
    ) != (
        expected_count,
        768,
    ):
        raise RuntimeError(
            f"{cache_name}: final embedding shape "
            f"is {tuple(all_embeddings.shape)}"
        )

    if not torch.isfinite(
        all_embeddings
    ).all():
        raise RuntimeError(
            f"{cache_name}: non-finite embeddings detected"
        )

    print("Shard count:", len(shard_paths))
    print("Question IDs:", all_ids.numel())
    print(
        "Embedding matrix:",
        tuple(all_embeddings.shape),
    )
    print(
        "Embedding dtype:",
        all_embeddings.dtype,
    )
    print("ID continuity: PASS")
    print("Shape validation: PASS")
    print("Finite-value validation: PASS")

    return all_embeddings


# ============================================================
# TARGET ENCODING
# ============================================================

BINARY_MAP = {
    value: index
    for index, value in enumerate(
        BINARY_CLASSES
    )
}

LAND_COVER_MAP = {
    value: index
    for index, value in enumerate(
        LAND_COVER_CLASSES
    )
}

RATIO_MAP = {
    value: index
    for index, value in enumerate(
        RATIO_CLASSES
    )
}

RATIO_TYPE_MAP = {
    value: index
    for index, value in enumerate(
        RATIO_TYPE_CLASSES
    )
}


def encode_answer(
    question_type: str,
    answer: str,
) -> tuple[str, int]:

    if question_type not in QUESTION_TYPE_TO_HEAD:
        raise ValueError(
            f"Unknown question type: "
            f"{question_type}"
        )

    head = QUESTION_TYPE_TO_HEAD[
        question_type
    ]

    if head == "binary":

        if answer not in BINARY_MAP:
            raise ValueError(
                f"Unexpected binary answer "
                f"'{answer}' for "
                f"{question_type}"
            )

        return head, BINARY_MAP[answer]

    if head == "land_cover":

        if answer not in LAND_COVER_MAP:
            raise ValueError(
                f"Unexpected land-cover answer "
                f"'{answer}' for "
                f"{question_type}"
            )

        return head, LAND_COVER_MAP[answer]

    if head == "ratio":

        if answer not in RATIO_MAP:
            raise ValueError(
                f"Unexpected ratio answer "
                f"'{answer}'"
            )

        return head, RATIO_MAP[answer]

    if head == "ratio_type":

        if answer not in RATIO_TYPE_MAP:
            raise ValueError(
                f"Unexpected ratio-type answer "
                f"'{answer}'"
            )

        return head, RATIO_TYPE_MAP[answer]

    raise RuntimeError(
        f"Unhandled head: {head}"
    )


# ============================================================
# QA DATASET
# ============================================================

class TemporalRAMDataset(Dataset):

    def __init__(
        self,
        manifest_path: Path,
        feature_store: TemporalFeatureStore,
        question_embeddings: torch.Tensor,
        split_name: str,
    ) -> None:

        self.manifest_path = Path(
            manifest_path
        )

        self.feature_store = feature_store
        self.question_embeddings = (
            question_embeddings
        )

        self.split_name = split_name

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found:\n"
                f"{self.manifest_path}"
            )

        self.df = pd.read_csv(
            self.manifest_path
        )

        required_columns = {
            "pair_id",
            "file_name",
            "question_id",
            "question_type",
            "question",
            "answer",
        }

        missing = (
            required_columns
            - set(self.df.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns in "
                f"{self.split_name} manifest: "
                f"{sorted(missing)}"
            )

        if self.df.empty:
            raise ValueError(
                f"{self.split_name} manifest is empty."
            )

        print()
        print(
            f"{self.split_name.capitalize()} QA rows:",
            len(self.df),
        )

        # ----------------------------------------------------
        # Precompute numeric arrays.
        # ----------------------------------------------------

        self.pair_ids = torch.tensor(
            self.df["pair_id"]
            .astype(int)
            .to_numpy(),
            dtype=torch.long,
        )

        self.question_ids = torch.tensor(
            self.df["question_id"]
            .astype(int)
            .to_numpy(),
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Validate pair IDs against feature store.
        # ----------------------------------------------------

        missing_pairs = sorted(
            set(
                self.pair_ids.tolist()
            )
            - set(
                feature_store.pair_id_to_index.keys()
            )
        )

        if missing_pairs:
            raise ValueError(
                f"{self.split_name}: "
                f"{len(missing_pairs)} pair IDs "
                f"missing from feature store. "
                f"Examples: {missing_pairs[:10]}"
            )

        # ----------------------------------------------------
        # Validate question IDs.
        # ----------------------------------------------------

        max_qid = int(
            self.question_ids.max()
        )

        if max_qid >= (
            question_embeddings.shape[0]
        ):
            raise ValueError(
                f"{self.split_name}: question ID "
                f"{max_qid} exceeds embedding cache "
                f"size {question_embeddings.shape[0]}"
            )

        # ----------------------------------------------------
        # Encode all targets once.
        # ----------------------------------------------------

        head_names = []
        target_indices = []

        for question_type, answer in zip(
            self.df["question_type"],
            self.df["answer"],
        ):

            head, target = encode_answer(
                str(question_type),
                str(answer),
            )

            head_names.append(head)
            target_indices.append(target)

        self.head_names = np.asarray(
            head_names
        )

        self.targets = torch.tensor(
            target_indices,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Validate exact type vocabulary coverage.
        # ----------------------------------------------------

        type_counts = (
            self.df["question_type"]
            .value_counts()
            .sort_index()
        )

        print(
            f"{self.split_name} question types:"
        )

        for question_type, count in (
            type_counts.items()
        ):
            print(
                f"  {question_type:20s} {count:7d}"
            )

        # ----------------------------------------------------
        # Verify every pair has a feature.
        # ----------------------------------------------------

        print(
            f"{self.split_name}: "
            "pair-feature mapping: PASS"
        )

        print(
            f"{self.split_name}: "
            "question-embedding mapping: PASS"
        )

        print(
            f"{self.split_name}: "
            "label mapping: PASS"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor | int | str]:

        pair_id = int(
            self.pair_ids[index]
        )

        question_id = int(
            self.question_ids[index]
        )

        feature_index = (
            self.feature_store.pair_id_to_index[
                pair_id
            ]
        )

        return {
            "pair_id": pair_id,
            "question_id": question_id,
            "question_type": str(
                self.df.iloc[index][
                    "question_type"
                ]
            ),
            "t1": self.feature_store.t1[
                feature_index
            ],
            "t2": self.feature_store.t2[
                feature_index
            ],
            "diff": self.feature_store.diff[
                feature_index
            ],
            "question": self.question_embeddings[
                question_id
            ],
            "target": self.targets[index],
        }


# ============================================================
# COLLATION
# ============================================================

def temporal_collate(
    batch,
):

    return {
        "pair_id": torch.tensor(
            [
                item["pair_id"]
                for item in batch
            ],
            dtype=torch.long,
        ),
        "question_id": torch.tensor(
            [
                item["question_id"]
                for item in batch
            ],
            dtype=torch.long,
        ),
        "question_type": [
            item["question_type"]
            for item in batch
        ],
        "t1": torch.stack(
            [
                item["t1"]
                for item in batch
            ],
            dim=0,
        ),
        "t2": torch.stack(
            [
                item["t2"]
                for item in batch
            ],
            dim=0,
        ),
        "diff": torch.stack(
            [
                item["diff"]
                for item in batch
            ],
            dim=0,
        ),
        "question": torch.stack(
            [
                item["question"]
                for item in batch
            ],
            dim=0,
        ),
        "target": torch.stack(
            [
                item["target"]
                for item in batch
            ],
            dim=0,
        ),
    }


# ============================================================
# ROUTED MULTI-HEAD LOSS
# ============================================================

def routed_loss(
    outputs,
    question_types,
    targets,
):
    """
    Apply CE only to the answer head appropriate for each
    temporal question type.

    Exact verified model interface:

        outputs["answer_logits"]["binary"]             [B,2]
        outputs["answer_logits"]["change_ratio"]       [B,11]
        outputs["answer_logits"]["change_ratio_types"] [B,9]
        outputs["answer_logits"]["land_cover"]         [B,6]

    outputs["change_logits"] is a spatial 14x14 change/evidence
    map and is NOT an answer-classification head.
    """

    answer_logits = outputs[
        "answer_logits"
    ]

    total_loss = torch.tensor(
        0.0,
        device=targets.device,
    )

    loss_count = 0

    per_head_loss = {}

    head_groups = defaultdict(list)

    for index, question_type in enumerate(
        question_types
    ):

        head = QUESTION_TYPE_TO_HEAD[
            question_type
        ]

        head_groups[head].append(index)

    # --------------------------------------------------------
    # Binary:
    #
    # change_or_not
    # increase_or_not
    # decrease_or_not
    # --------------------------------------------------------

    if "binary" in head_groups:

        indices = torch.tensor(
            head_groups["binary"],
            dtype=torch.long,
            device=targets.device,
        )

        logits = answer_logits[
            "binary"
        ][indices]

        y = targets[
            indices
        ]

        loss = F.cross_entropy(
            logits,
            y,
        )

        total_loss = total_loss + loss
        loss_count += 1

        per_head_loss[
            "binary"
        ] = float(loss.detach())

    # --------------------------------------------------------
    # Land cover:
    #
    # smallest_change
    # largest_change
    # change_to_what
    # --------------------------------------------------------

    if "land_cover" in head_groups:

        indices = torch.tensor(
            head_groups["land_cover"],
            dtype=torch.long,
            device=targets.device,
        )

        logits = answer_logits[
            "land_cover"
        ][indices]

        y = targets[
            indices
        ]

        loss = F.cross_entropy(
            logits,
            y,
        )

        total_loss = total_loss + loss
        loss_count += 1

        per_head_loss[
            "land_cover"
        ] = float(loss.detach())

    # --------------------------------------------------------
    # Change ratio:
    #
    # change_ratio
    # --------------------------------------------------------

    if "ratio" in head_groups:

        indices = torch.tensor(
            head_groups["ratio"],
            dtype=torch.long,
            device=targets.device,
        )

        logits = answer_logits[
            "change_ratio"
        ][indices]

        y = targets[
            indices
        ]

        loss = F.cross_entropy(
            logits,
            y,
        )

        total_loss = total_loss + loss
        loss_count += 1

        per_head_loss[
            "ratio"
        ] = float(loss.detach())

    # --------------------------------------------------------
    # Change ratio types:
    #
    # change_ratio_types
    # --------------------------------------------------------

    if "ratio_type" in head_groups:

        indices = torch.tensor(
            head_groups["ratio_type"],
            dtype=torch.long,
            device=targets.device,
        )

        logits = answer_logits[
            "change_ratio_types"
        ][indices]

        y = targets[
            indices
        ]

        loss = F.cross_entropy(
            logits,
            y,
        )

        total_loss = total_loss + loss
        loss_count += 1

        per_head_loss[
            "ratio_type"
        ] = float(loss.detach())

    if loss_count == 0:

        raise RuntimeError(
            "No valid question types in batch."
        )

    total_loss = (
        total_loss / loss_count
    )

    return total_loss, per_head_loss
# ============================================================
# OUTPUT-HEAD ACCESS
# ============================================================

def get_head_logits(
    outputs,
    question_type: str,
):
    """
    Return the exact verified answer-classification tensor
    for the supplied temporal question type.
    """

    answer_logits = outputs[
        "answer_logits"
    ]

    head = QUESTION_TYPE_TO_HEAD[
        question_type
    ]

    if head == "binary":

        return answer_logits[
            "binary"
        ]

    if head == "ratio":

        return answer_logits[
            "change_ratio"
        ]

    if head == "ratio_type":

        return answer_logits[
            "change_ratio_types"
        ]

    if head == "land_cover":

        return answer_logits[
            "land_cover"
        ]

    raise RuntimeError(
        f"Unknown answer head: {head}"
    )

# ============================================================
# PREDICTION / ACCURACY
# ============================================================

def update_accuracy_stats(
    outputs,
    question_types,
    targets,
    stats,
):

    for index, question_type in enumerate(
        question_types
    ):

        logits = get_head_logits(
            outputs,
            question_type,
        )[index:index + 1]

        predicted = int(
            torch.argmax(
                logits,
                dim=-1,
            )[0]
        )

        target = int(
            targets[index]
        )

        stats[
            question_type
        ]["total"] += 1

        if predicted == target:
            stats[
                question_type
            ]["correct"] += 1


def accuracy_summary(
    stats,
):

    total_correct = 0
    total_count = 0

    result = {}

    for question_type in sorted(
        stats.keys()
    ):

        correct = stats[
            question_type
        ]["correct"]

        total = stats[
            question_type
        ]["total"]

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        result[
            question_type
        ] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }

        total_correct += correct
        total_count += total

    result["overall"] = {
        "correct": total_correct,
        "total": total_count,
        "accuracy": (
            total_correct / total_count
            if total_count > 0
            else 0.0
        ),
    }

    return result


# ============================================================
# MODEL FORWARD COMPATIBILITY
# ============================================================

def forward_model(
    model,
    t1,
    t2,
    diff,
    question,
):
    """
    Use the exact verified TemporalChangeVQA interface.

    Verified inputs:

        t1       [B,196,768]
        t2       [B,196,768]
        diff     [B,196,768]
        question [B,768]
    """

    return model(
        t1,
        t2,
        diff,
        question,
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
):

    model.train()

    running_loss = 0.0
    examples = 0

    stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    epoch_start = time.perf_counter()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):

        t1 = batch["t1"].to(
            DEVICE,
            non_blocking=True,
        )

        t2 = batch["t2"].to(
            DEVICE,
            non_blocking=True,
        )

        diff = batch["diff"].to(
            DEVICE,
            non_blocking=True,
        )

        question = batch[
            "question"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        targets = batch[
            "target"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                DEVICE.type == "cuda"
            ),
        ):

            outputs = forward_model(
                model,
                t1,
                t2,
                diff,
                question,
            )

            loss, _ = routed_loss(
                outputs,
                batch["question_type"],
                targets,
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        batch_size = targets.shape[0]

        running_loss += (
            float(loss.detach())
            * batch_size
        )

        examples += batch_size

        # Accuracy is computed without gradients.
        with torch.no_grad():

            update_accuracy_stats(
                outputs,
                batch["question_type"],
                targets,
                stats,
            )

        if (
            batch_index == 1
            or batch_index % 250 == 0
            or batch_index == len(loader)
        ):

            elapsed = (
                time.perf_counter()
                - epoch_start
            )

            examples_per_second = (
                examples / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"  Batch "
                f"{batch_index:5d}/"
                f"{len(loader):5d} "
                f"| loss={float(loss.detach()):.4f} "
                f"| ex/s={examples_per_second:.2f}"
            )

    epoch_loss = (
        running_loss / examples
    )

    metrics = accuracy_summary(
        stats
    )

    return epoch_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
):

    model.eval()

    running_loss = 0.0
    examples = 0

    stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    start_time = time.perf_counter()

    for batch in loader:

        t1 = batch["t1"].to(
            DEVICE,
            non_blocking=True,
        )

        t2 = batch["t2"].to(
            DEVICE,
            non_blocking=True,
        )

        diff = batch["diff"].to(
            DEVICE,
            non_blocking=True,
        )

        question = batch[
            "question"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        targets = batch[
            "target"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                DEVICE.type == "cuda"
            ),
        ):

            outputs = forward_model(
                model,
                t1,
                t2,
                diff,
                question,
            )

            loss, _ = routed_loss(
                outputs,
                batch["question_type"],
                targets,
            )

        batch_size = targets.shape[0]

        running_loss += (
            float(loss)
            * batch_size
        )

        examples += batch_size

        update_accuracy_stats(
            outputs,
            batch["question_type"],
            targets,
            stats,
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    epoch_loss = (
        running_loss / examples
    )

    metrics = accuracy_summary(
        stats
    )

    return (
        epoch_loss,
        metrics,
        elapsed,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    history,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "model_class": "TemporalChangeVQA",
        "model_parameters": sum(
            p.numel()
            for p in model.parameters()
        ),
        "history": history,
    }

    torch.save(
        payload,
        path,
    )


# ============================================================
# HISTORY
# ============================================================

def save_history(
    history,
):

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_change_or_not",
        "val_increase_or_not",
        "val_decrease_or_not",
        "val_smallest_change",
        "val_largest_change",
        "val_change_to_what",
        "val_change_ratio",
        "val_change_ratio_types",
        "epoch_time_seconds",
        "validation_time_seconds",
    ]

    with HISTORY_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in history:

            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SATQUERYAI — TEMPORAL CHANGE VQA TRAINING")
    print("=" * 72)

    print()
    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Patience:", PATIENCE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Seed:", SEED)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        print(
            "CUDA:",
            torch.version.cuda,
        )

    # --------------------------------------------------------
    # Feature store.
    # --------------------------------------------------------

    feature_store = (
        TemporalFeatureStore(
            FEATURE_DIR
        )
    )

    # --------------------------------------------------------
    # Question caches.
    # --------------------------------------------------------

    train_questions = (
        load_question_embeddings(
            TRAIN_QUESTION_CACHE,
            expected_count=65967,
            cache_name="training question cache",
        )
    )

    val_questions = (
        load_question_embeddings(
            VAL_QUESTION_CACHE,
            expected_count=16441,
            cache_name="validation question cache",
        )
    )

    # --------------------------------------------------------
    # Datasets.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("BUILDING RAM-BACKED DATASETS")
    print("=" * 72)

    train_dataset = (
        TemporalRAMDataset(
            TRAIN_MANIFEST,
            feature_store,
            train_questions,
            "train",
        )
    )

    val_dataset = (
        TemporalRAMDataset(
            VAL_MANIFEST,
            feature_store,
            val_questions,
            "validation",
        )
    )

    # --------------------------------------------------------
    # DataLoaders.
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=temporal_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=temporal_collate,
    )

    print()
    print(
        "Train batches per epoch:",
        len(train_loader),
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("BUILDING FROZEN TEMPORAL ARCHITECTURE")
    print("=" * 72)

    model = TemporalChangeVQA()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    if parameter_count != 1_744_925:

        raise RuntimeError(
            "Temporal model parameter count "
            "changed from the verified value. "
            f"Expected 1,744,925; got "
            f"{parameter_count}."
        )

    print(
        "Architecture parameter count: PASS"
    )

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Optimizer.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=(
                DEVICE.type == "cuda"
            )
        )
        if DEVICE.type == "cuda"
        else None
    )

    # --------------------------------------------------------
    # Initial forward verification.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("INITIAL REAL-DATA FORWARD CHECK")
    print("=" * 72)

    first_batch = next(
        iter(train_loader)
    )

    test_t1 = first_batch[
        "t1"
    ].to(
        DEVICE
    )

    test_t2 = first_batch[
        "t2"
    ].to(
        DEVICE
    )

    test_diff = first_batch[
        "diff"
    ].to(
        DEVICE
    )

    test_question = first_batch[
        "question"
    ].to(
        DEVICE
    )

    with torch.no_grad():

        outputs = forward_model(
            model,
            test_t1,
            test_t2,
            test_diff,
            test_question,
        )

    expected_core_shapes = {
        "query_attention": (
            BATCH_SIZE,
            14,
            14,
        ),
        "change_logits": (
            BATCH_SIZE,
            14,
            14,
        ),
        "fused_tokens": (
            BATCH_SIZE,
            196,
            256,
        ),
        "joint": (
            BATCH_SIZE,
            256,
        ),
    }

    expected_answer_shapes = {
        "binary": (
            BATCH_SIZE,
            2,
        ),
        "change_ratio": (
            BATCH_SIZE,
            11,
        ),
        "change_ratio_types": (
            BATCH_SIZE,
            9,
        ),
        "land_cover": (
            BATCH_SIZE,
            6,
        ),
    }

    for key, expected in (
        expected_core_shapes.items()
    ):

        if key not in outputs:

            raise RuntimeError(
                f"Expected output '{key}' "
                f"not found."
            )

        actual = tuple(
            outputs[key].shape
        )

        print(
            f"{key:28s}: "
            f"{actual}"
        )

        if actual != expected:

            raise RuntimeError(
                f"{key}: expected "
                f"{expected}, got {actual}"
            )


    if "answer_logits" not in outputs:

        raise RuntimeError(
            "Missing nested answer_logits output."
        )


    if not isinstance(
        outputs["answer_logits"],
        dict,
    ):

        raise RuntimeError(
            "answer_logits must be a dict."
        )


    for key, expected in (
        expected_answer_shapes.items()
    ):

        if key not in outputs[
            "answer_logits"
        ]:

            raise RuntimeError(
                f"Missing answer head: {key}"
            )

        actual = tuple(
            outputs[
                "answer_logits"
            ][key].shape
        )

        print(
            f"answer_logits[{key:18s}]: "
            f"{actual}"
        )

        if actual != expected:

            raise RuntimeError(
                f"answer_logits[{key}]: "
                f"expected {expected}, "
                f"got {actual}"
            )

    print(
        "Real-data forward check: PASS"
    )
    
    # --------------------------------------------------------
    # Training loop.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("STARTING TRAINING")
    print("=" * 72)

    history = []

    best_val_loss = float(
        "inf"
    )

    best_epoch = None
    epochs_without_improvement = 0

    training_start = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print("=" * 72)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 72)

        epoch_start = (
            time.perf_counter()
        )

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
            )
        )

        val_loss, val_metrics, val_time = (
            validate(
                model,
                val_loader,
            )
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        train_accuracy = (
            train_metrics[
                "overall"
            ]["accuracy"]
        )

        val_accuracy = (
            val_metrics[
                "overall"
            ]["accuracy"]
        )

        print()
        print(
            f"Train loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy:.6f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.6f}"
        )

        print(
            f"Validation accuracy: "
            f"{val_accuracy:.6f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.2f}s"
        )

        print(
            f"Validation time: "
            f"{val_time:.2f}s"
        )

        print()
        print("Validation accuracy by question type:")

        for question_type in sorted(
            val_metrics.keys()
        ):

            if question_type == "overall":
                continue

            metric = val_metrics[
                question_type
            ]

            print(
                f"  "
                f"{question_type:20s} "
                f"{metric['correct']:5d}/"
                f"{metric['total']:5d} "
                f"="
                f"{metric['accuracy']:.4f}"
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_change_or_not": (
                val_metrics["change_or_not"][
                    "accuracy"
                ]
            ),
            "val_increase_or_not": (
                val_metrics["increase_or_not"][
                    "accuracy"
                ]
            ),
            "val_decrease_or_not": (
                val_metrics["decrease_or_not"][
                    "accuracy"
                ]
            ),
            "val_smallest_change": (
                val_metrics["smallest_change"][
                    "accuracy"
                ]
            ),
            "val_largest_change": (
                val_metrics["largest_change"][
                    "accuracy"
                ]
            ),
            "val_change_to_what": (
                val_metrics["change_to_what"][
                    "accuracy"
                ]
            ),
            "val_change_ratio": (
                val_metrics["change_ratio"][
                    "accuracy"
                ]
            ),
            "val_change_ratio_types": (
                val_metrics["change_ratio_types"][
                    "accuracy"
                ]
            ),
            "epoch_time_seconds": epoch_time,
            "validation_time_seconds": val_time,
        }

        history.append(
            row
        )

        # ----------------------------------------------------
        # Save latest.
        # ----------------------------------------------------

        latest_path = (
            OUTPUT_DIR
            / "latest.pt"
        )

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            val_accuracy,
            history,
        )

        # ----------------------------------------------------
        # Best validation checkpoint.
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_path = (
                OUTPUT_DIR
                / "best.pt"
            )

            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                val_accuracy,
                history,
            )

            print()
            print(
                "NEW BEST CHECKPOINT"
            )

            print(
                "Best validation loss:",
                f"{best_val_loss:.6f}",
            )

            print(
                "Checkpoint:",
                best_path,
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "No validation-loss improvement."
            )

            print(
                "Patience:",
                f"{epochs_without_improvement}/"
                f"{PATIENCE}",
            )

        save_history(
            history
        )

        # ----------------------------------------------------
        # Early stopping.
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    summary = {
        "device": str(DEVICE),
        "gpu": (
            torch.cuda.get_device_name(0)
            if DEVICE.type == "cuda"
            else None
        ),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history),
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "train_pairs": 1600,
        "validation_pairs": 400,
        "test_pairs_reserved": 968,
        "train_qa_rows": len(
            train_dataset
        ),
        "validation_qa_rows": len(
            val_dataset
        ),
        "model": "TemporalChangeVQA",
        "trainable_parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_val_loss
            if best_epoch is not None
            else None
        ),
        "total_training_seconds": (
            total_training_time
        ),
        "best_checkpoint": str(
            OUTPUT_DIR
            / "best.pt"
        ),
        "latest_checkpoint": str(
            OUTPUT_DIR
            / "latest.pt"
        ),
    }

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("TEMPORAL TRAINING COMPLETE")
    print("=" * 72)

    print(
        "Epochs completed:",
        len(history),
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation loss:",
        best_val_loss,
    )

    print(
        "Total training time:",
        f"{total_training_time:.2f}s",
    )

    print()
    print(
        "Best checkpoint:",
        OUTPUT_DIR / "best.pt",
    )

    print(
        "Training history:",
        HISTORY_FILE,
    )

    print(
        "Summary:",
        SUMMARY_FILE,
    )


if __name__ == "__main__":
    main()