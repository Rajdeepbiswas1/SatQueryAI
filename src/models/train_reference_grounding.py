import os
import sys
import random
import re

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from src.models.grounding_heatmap_model import GroundingHeatmapModel


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PARQUET = (
    r"D:\SatQueryAI\data\processed"
    r"\reference_grounding_split"
    r"\reference_train.parquet"
)

VAL_PARQUET = (
    r"D:\SatQueryAI\data\processed"
    r"\reference_grounding_split"
    r"\reference_val.parquet"
)

TEXT_CACHE_DIR = (
    r"D:\SatQueryAI\data\processed"
    r"\grounding_text_embeddings"
    r"\grounding_text_embedding_shards"
)

DOFA_ROOT = (
    r"D:\SatQueryAI\data\processed"
    r"\dofa_features_download"
    r"\dofa_features"
)

CHECKPOINT_DIR = (
    r"D:\SatQueryAI\models\checkpoints"
    r"\reference_grounding_large"
)

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "best.pt",
)

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "last.pt",
)

HISTORY_PATH = os.path.join(
    CHECKPOINT_DIR,
    "training_history.csv",
)

SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 25

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device("cpu")

QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90

HEATMAP_WEIGHT = 1.0

# Small auxiliary contribution only.
# The final box is still obtained from the heatmap.
BOX_WEIGHT = 0.10


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ============================================================
# IOU
# ============================================================

def box_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        *
        max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        *
        max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


# ============================================================
# HEATMAP → BOX
# ============================================================

def quantile_from_distribution(values, q):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.maximum(
        values,
        0.0,
    )

    total = values.sum()

    if total <= 0.0:
        return 0.5

    values = values / total

    cdf = np.cumsum(values)

    index = np.searchsorted(
        cdf,
        q,
        side="left",
    )

    index = max(
        0,
        min(
            index,
            len(values) - 1,
        ),
    )

    return float(
        (index + 0.5)
        /
        len(values)
    )


def heatmap_to_box(
    heatmap,
    low=QUANTILE_LOW,
    high=QUANTILE_HIGH,
):

    heatmap = np.asarray(
        heatmap,
        dtype=np.float64,
    )

    heatmap = np.maximum(
        heatmap,
        0.0,
    )

    total = heatmap.sum()

    if total <= 0.0:
        return [
            0.0,
            0.0,
            1.0,
            1.0,
        ]

    heatmap = (
        heatmap
        /
        total
    )

    x_distribution = heatmap.sum(
        axis=0
    )

    y_distribution = heatmap.sum(
        axis=1
    )

    x1 = quantile_from_distribution(
        x_distribution,
        low,
    )

    x2 = quantile_from_distribution(
        x_distribution,
        high,
    )

    y1 = quantile_from_distribution(
        y_distribution,
        low,
    )

    y2 = quantile_from_distribution(
        y_distribution,
        high,
    )

    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    ]


# ============================================================
# DOFA CACHE
# ============================================================

class DofaShardCache:

    def __init__(self, directory):

        self.directory = directory

        self.files = sorted(
            [
                os.path.join(
                    directory,
                    filename,
                )
                for filename in os.listdir(
                    directory
                )
                if filename.endswith(".pt")
            ]
        )

        if not self.files:
            raise RuntimeError(
                f"No DOFA shards found in:\n{directory}"
            )

        self.index = {}
        self.shards = {}

        print(
            f"Found {len(self.files)} cache shards "
            f"in {directory}"
        )

        for shard_id, filepath in enumerate(
            self.files
        ):

            data = torch.load(
                filepath,
                map_location="cpu",
            )

            if not isinstance(data, dict):
                raise RuntimeError(
                    f"Unexpected DOFA shard format:\n"
                    f"{filepath}"
                )

            required = [
                "patch_ids",
                "s1",
                "s2",
            ]

            for key in required:
                if key not in data:
                    raise RuntimeError(
                        f"Missing '{key}' in:\n"
                        f"{filepath}"
                    )

            self.shards[shard_id] = data

            patch_ids = data["patch_ids"]

            for local_index, patch_id in enumerate(
                patch_ids
            ):

                if torch.is_tensor(patch_id):
                    patch_id = patch_id.item()

                patch_id = str(patch_id)

                if patch_id in self.index:
                    raise RuntimeError(
                        f"Duplicate patch ID in DOFA cache:\n"
                        f"{patch_id}"
                    )

                self.index[patch_id] = (
                    shard_id,
                    local_index,
                )

        print(
            f"Indexed {len(self.index):,} "
            f"cached patch IDs"
        )

    def get(self, patch_id):

        patch_id = str(patch_id)

        if patch_id not in self.index:
            raise KeyError(
                f"Patch not found in DOFA cache:\n"
                f"{patch_id}"
            )

        shard_id, local_index = (
            self.index[patch_id]
        )

        shard = self.shards[
            shard_id
        ]

        return (
            shard["s1"][local_index],
            shard["s2"][local_index],
        )


# ============================================================
# TEXT EMBEDDING CACHE
# ============================================================

class GroundingTextCache:

    def __init__(self, directory):

        self.directory = directory

        self.files = sorted(
            [
                os.path.join(
                    directory,
                    filename,
                )
                for filename in os.listdir(
                    directory
                )
                if filename.endswith(".pt")
            ]
        )

        if not self.files:
            raise RuntimeError(
                f"No grounding text shards found in:\n"
                f"{directory}"
            )

        self.index = {}
        self.shards = {}

        print(
            f"Found {len(self.files)} text embedding shards"
        )

        for shard_id, filepath in enumerate(
            self.files
        ):

            data = torch.load(
                filepath,
                map_location="cpu",
            )

            if not isinstance(data, dict):
                raise RuntimeError(
                    f"Unexpected text shard format:\n"
                    f"{filepath}"
                )

            # ------------------------------------------------
            # This is the VERIFIED shard schema:
            #
            # source_row_index
            # patch_ids
            # text_embeddings
            # points
            # target_boxes
            # modes
            # categories
            # ------------------------------------------------

            required = [
                "source_row_index",
                "text_embeddings",
            ]

            for key in required:
                if key not in data:
                    raise RuntimeError(
                        f"Missing '{key}' in:\n"
                        f"{filepath}"
                    )

            source_indices = data[
                "source_row_index"
            ]

            embeddings = data[
                "text_embeddings"
            ]

            if not torch.is_tensor(
                source_indices
            ):
                raise RuntimeError(
                    f"'source_row_index' is not a tensor:\n"
                    f"{filepath}"
                )

            if not torch.is_tensor(
                embeddings
            ):
                raise RuntimeError(
                    f"'text_embeddings' is not a tensor:\n"
                    f"{filepath}"
                )

            if (
                embeddings.ndim != 2
                or embeddings.shape[1] != 768
            ):
                raise RuntimeError(
                    f"Unexpected text embedding shape "
                    f"{embeddings.shape} in:\n"
                    f"{filepath}"
                )

            if (
                len(source_indices)
                != len(embeddings)
            ):
                raise RuntimeError(
                    "source_row_index and "
                    "text_embeddings lengths differ "
                    f"in:\n{filepath}"
                )

            self.shards[shard_id] = data

            for local_index, row_index in enumerate(
                source_indices.tolist()
            ):

                row_index = int(
                    row_index
                )

                if row_index in self.index:
                    raise RuntimeError(
                        "Duplicate source_row_index "
                        f"{row_index} across text cache shards."
                    )

                self.index[row_index] = (
                    shard_id,
                    local_index,
                )

        print(
            f"Indexed {len(self.index):,} "
            f"text embeddings"
        )

    def get(self, source_row_index):

        source_row_index = int(
            source_row_index
        )

        if (
            source_row_index
            not in self.index
        ):
            raise KeyError(
                "Text embedding missing for "
                f"source row {source_row_index}"
            )

        shard_id, local_index = (
            self.index[
                source_row_index
            ]
        )

        return (
            self.shards[
                shard_id
            ]["text_embeddings"][
                local_index
            ].float()
        )


# ============================================================
# REFERENCE GROUNDING DATASET
# ============================================================

class ReferenceGroundingDataset(
    Dataset
):

    def __init__(
        self,
        parquet_path,
        dofa_cache,
        text_cache,
    ):

        self.df = pd.read_parquet(
            parquet_path
        )

        self.dofa_cache = (
            dofa_cache
        )

        self.text_cache = (
            text_cache
        )

        print(
            f"Loaded {len(self.df):,} rows from:"
            f"\n  {parquet_path}"
        )

        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required = [
            "ID",
            "patch_id",
            "input",
            "output",
            "type",
            "category",
        ]

        missing = [
            column
            for column in required
            if column not in self.df.columns
        ]

        if missing:
            raise RuntimeError(
                "Missing dataset columns: "
                + ", ".join(missing)
            )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        type_ok = (
            self.df["type"]
            .astype(str)
            .str.lower()
            .eq("bounding box")
        )

        category_ok = (
            self.df["category"]
            .astype(str)
            .str.lower()
            .eq("reference")
        )

        if not type_ok.all():
            raise RuntimeError(
                "Training/validation parquet contains "
                "non-bounding-box examples."
            )

        if not category_ok.all():
            raise RuntimeError(
                "Training/validation parquet contains "
                "non-reference examples."
            )

        # ----------------------------------------------------
        # Build stable source-row mapping.
        #
        # The split generator preserved the original
        # annotation row under original_row_index.
        # ----------------------------------------------------

        if (
            "original_row_index"
            not in self.df.columns
        ):
            raise RuntimeError(
                "original_row_index is missing from "
                f"{parquet_path}.\n"
                "The split must be regenerated with "
                "the original annotation index preserved."
            )

        self.df[
            "original_row_index"
        ] = self.df[
            "original_row_index"
        ].astype(int)

        # ----------------------------------------------------
        # Validate every row has a cached embedding
        # before training starts.
        # ----------------------------------------------------

        missing_embeddings = []

        for row_index in self.df[
            "original_row_index"
        ].tolist():

            if (
                row_index
                not in self.text_cache.index
            ):

                missing_embeddings.append(
                    row_index
                )

                if len(
                    missing_embeddings
                ) >= 10:
                    break

        if missing_embeddings:
            raise RuntimeError(
                "Missing text embeddings for "
                "source rows: "
                + str(missing_embeddings)
            )

    def __len__(self):
        return len(self.df)

    def parse_box(self, value):

        if isinstance(
            value,
            str,
        ):

            numbers = re.findall(
                r"-?(?:\d+(?:\.\d*)?|\.\d+)",
                value,
            )

            if len(numbers) < 4:
                raise RuntimeError(
                    f"Could not parse target box:\n"
                    f"{value}"
                )

            box = [
                float(numbers[0]),
                float(numbers[1]),
                float(numbers[2]),
                float(numbers[3]),
            ]

        else:

            box = np.asarray(
                value,
                dtype=np.float32,
            ).reshape(-1).tolist()

            if len(box) < 4:
                raise RuntimeError(
                    f"Invalid target box:\n"
                    f"{value}"
                )

            box = box[:4]

        x1, y1, x2, y2 = box

        x1 = float(
            np.clip(x1, 0.0, 1.0)
        )

        y1 = float(
            np.clip(y1, 0.0, 1.0)
        )

        x2 = float(
            np.clip(x2, 0.0, 1.0)
        )

        y2 = float(
            np.clip(y2, 0.0, 1.0)
        )

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        return torch.tensor(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dtype=torch.float32,
        )

    def extract_phrase(self, text):

        if text is None:
            return ""

        text = str(text)

        match = re.search(
            r"<ref>(.*?)</ref>",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

        return text.strip()

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        patch_id = str(
            row["patch_id"]
        )

        source_row_index = int(
            row["original_row_index"]
        )

        # ----------------------------------------------------
        # DOFA features
        # ----------------------------------------------------

        s1, s2 = (
            self.dofa_cache.get(
                patch_id
            )
        )

        # ----------------------------------------------------
        # Text embedding
        # ----------------------------------------------------

        text_embedding = (
            self.text_cache.get(
                source_row_index
            )
        )

        # ----------------------------------------------------
        # Target box
        # ----------------------------------------------------

        target_box = self.parse_box(
            row["output"]
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        phrase = self.extract_phrase(
            row["input"]
        )

        return {
            "s1": s1.float(),
            "s2": s2.float(),
            "text_embedding":
                text_embedding.float(),
            "target_box":
                target_box,
            "patch_id":
                patch_id,
            "row_index":
                source_row_index,
            "category":
                "reference",
            "phrase":
                phrase,
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_reference_batch(
    batch
):

    return {
        "s1": torch.stack(
            [
                item["s1"]
                for item in batch
            ]
        ),

        "s2": torch.stack(
            [
                item["s2"]
                for item in batch
            ]
        ),

        "text_embedding": torch.stack(
            [
                item["text_embedding"]
                for item in batch
            ]
        ),

        "target_box": torch.stack(
            [
                item["target_box"]
                for item in batch
            ]
        ),

        "patch_id": [
            item["patch_id"]
            for item in batch
        ],

        "row_index": [
            item["row_index"]
            for item in batch
        ],

        "category": [
            item["category"]
            for item in batch
        ],

        "phrase": [
            item["phrase"]
            for item in batch
        ],
    }


# ============================================================
# HEATMAP TARGET
# ============================================================

def create_box_overlap_target(
    boxes,
    height=14,
    width=14,
):

    batch_size = boxes.shape[0]

    targets = torch.zeros(
        (
            batch_size,
            height,
            width,
        ),
        dtype=torch.float32,
    )

    cell_w = 1.0 / width
    cell_h = 1.0 / height

    for b in range(
        batch_size
    ):

        x1, y1, x2, y2 = (
            boxes[b]
        )

        x1 = float(
            torch.clamp(
                x1,
                0.0,
                1.0,
            )
        )

        y1 = float(
            torch.clamp(
                y1,
                0.0,
                1.0,
            )
        )

        x2 = float(
            torch.clamp(
                x2,
                0.0,
                1.0,
            )
        )

        y2 = float(
            torch.clamp(
                y2,
                0.0,
                1.0,
            )
        )

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        total = 0.0

        for row in range(
            height
        ):

            cy1 = (
                row
                *
                cell_h
            )

            cy2 = (
                row + 1
            ) * cell_h

            iy1 = max(
                y1,
                cy1,
            )

            iy2 = min(
                y2,
                cy2,
            )

            ih = max(
                0.0,
                iy2 - iy1,
            )

            if ih <= 0.0:
                continue

            for col in range(
                width
            ):

                cx1 = (
                    col
                    *
                    cell_w
                )

                cx2 = (
                    col + 1
                ) * cell_w

                ix1 = max(
                    x1,
                    cx1,
                )

                ix2 = min(
                    x2,
                    cx2,
                )

                iw = max(
                    0.0,
                    ix2 - ix1,
                )

                overlap = (
                    iw * ih
                )

                if overlap > 0.0:

                    targets[
                        b,
                        row,
                        col,
                    ] = overlap

                    total += overlap

        if total > 0.0:

            targets[
                b
            ] /= total

        else:

            cx = (
                x1 + x2
            ) / 2.0

            cy = (
                y1 + y2
            ) / 2.0

            col = min(
                width - 1,
                max(
                    0,
                    int(
                        cx * width
                    ),
                ),
            )

            row = min(
                height - 1,
                max(
                    0,
                    int(
                        cy * height
                    ),
                ),
            )

            targets[
                b,
                row,
                col,
            ] = 1.0

    return targets


# ============================================================
# HEATMAP LOSS
# ============================================================

def heatmap_loss(
    predicted,
    target,
):

    predicted = predicted.clamp_min(
        1e-8
    )

    target = target.clamp_min(
        0.0
    )

    target_sum = target.sum(
        dim=(1, 2),
        keepdim=True,
    )

    target = (
        target
        /
        target_sum.clamp_min(
            1e-8
        )
    )

    loss = -(
        target
        *
        torch.log(
            predicted
        )
    ).sum(
        dim=(1, 2)
    )

    return loss.mean()


# ============================================================
# MODEL OUTPUT
# ============================================================

def extract_heatmap(output):

    if isinstance(
        output,
        dict,
    ):

        if "heatmap" in output:
            return output[
                "heatmap"
            ]

        if "spatial_heatmap" in output:
            return output[
                "spatial_heatmap"
            ]

    if torch.is_tensor(output):
        return output

    raise RuntimeError(
        "Model output does not contain a heatmap."
    )


def extract_box(output):

    if not isinstance(
        output,
        dict,
    ):
        return None

    possible_keys = [
        "box",
        "pred_box",
        "box_pred",
        "box_logits",
    ]

    for key in possible_keys:

        if key not in output:
            continue

        value = output[key]

        if (
            torch.is_tensor(value)
            and value.shape[-1] == 4
        ):

            return value

    return None


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    epoch,
):

    model.train()

    total_loss = 0.0
    total_hm_loss = 0.0
    total_box_loss = 0.0

    batches = 0

    for batch_idx, batch in enumerate(
        loader
    ):

        s1 = batch[
            "s1"
        ].to(
            DEVICE
        )

        s2 = batch[
            "s2"
        ].to(
            DEVICE
        )

        text_embedding = batch[
            "text_embedding"
        ].to(
            DEVICE
        )

        target_box = batch[
            "target_box"
        ].to(
            DEVICE
        )

        optimizer.zero_grad()

        output = model(
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
        )

        heatmap = extract_heatmap(
            output
        )

        if heatmap.dim() == 4:
            heatmap = heatmap.squeeze(
                1
            )

        if heatmap.dim() != 3:
            raise RuntimeError(
                f"Unexpected heatmap shape: "
                f"{heatmap.shape}"
            )

        target_heatmap = (
            create_box_overlap_target(
                target_box,
                height=heatmap.shape[-2],
                width=heatmap.shape[-1],
            )
            .to(DEVICE)
        )

        hm_loss = heatmap_loss(
            heatmap,
            target_heatmap,
        )

        aux_box = extract_box(
            output
        )

        if aux_box is not None:

            aux_box = torch.clamp(
                aux_box,
                0.0,
                1.0,
            )

            box_loss = F.smooth_l1_loss(
                aux_box,
                target_box,
            )

        else:

            box_loss = torch.tensor(
                0.0,
                device=DEVICE,
            )

        loss = (
            HEATMAP_WEIGHT
            *
            hm_loss
            +
            BOX_WEIGHT
            *
            box_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += float(
            loss.item()
        )

        total_hm_loss += float(
            hm_loss.item()
        )

        total_box_loss += float(
            box_loss.item()
        )

        batches += 1

        if (
            batch_idx + 1
        ) % 50 == 0:

            print(
                f"  Epoch {epoch:02d} | "
                f"Batch {batch_idx + 1:04d}/"
                f"{len(loader):04d} | "
                f"Loss {loss.item():.4f}"
            )

    return {
        "loss":
            total_loss / max(
                1,
                batches,
            ),

        "heatmap_loss":
            total_hm_loss / max(
                1,
                batches,
            ),

        "box_loss":
            total_box_loss / max(
                1,
                batches,
            ),
    }


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
):

    model.eval()

    ious = []

    records = []

    with torch.no_grad():

        for batch in loader:

            s1 = batch[
                "s1"
            ].to(
                DEVICE
            )

            s2 = batch[
                "s2"
            ].to(
                DEVICE
            )

            text_embedding = batch[
                "text_embedding"
            ].to(
                DEVICE
            )

            target_boxes = batch[
                "target_box"
            ]

            output = model(
                s1=s1,
                s2=s2,
                text_embedding=text_embedding,
            )

            heatmaps = extract_heatmap(
                output
            )

            if heatmaps.dim() == 4:
                heatmaps = heatmaps.squeeze(
                    1
                )

            for i in range(
                heatmaps.shape[0]
            ):

                prediction = (
                    heatmap_to_box(
                        heatmaps[i]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                )

                target = (
                    target_boxes[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64
                    )
                    .tolist()
                )

                iou = box_iou(
                    prediction,
                    target,
                )

                ious.append(
                    iou
                )

                records.append(
                    {
                        "row_index":
                            batch["row_index"][i],

                        "patch_id":
                            batch["patch_id"][i],

                        "phrase":
                            batch["phrase"][i],

                        "prediction":
                            prediction,

                        "target":
                            target,

                        "iou":
                            iou,
                    }
                )

    ious = np.asarray(
        ious,
        dtype=np.float64,
    )

    return {
        "mean_iou":
            float(
                ious.mean()
            ),

        "median_iou":
            float(
                np.median(ious)
            ),

        "iou_50":
            float(
                (ious >= 0.50).mean()
            ),

        "iou_25":
            float(
                (ious >= 0.25).mean()
            ),

        "ious":
            ious,

        "records":
            records,
    }


# ============================================================
# PER-PHRASE STATISTICS
# ============================================================

def calculate_phrase_statistics(
    records
):

    grouped = {}

    for record in records:

        phrase = record[
            "phrase"
        ]

        if phrase not in grouped:
            grouped[phrase] = []

        grouped[phrase].append(
            record["iou"]
        )

    results = []

    for phrase, values in grouped.items():

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        results.append(
            {
                "phrase":
                    phrase,

                "count":
                    len(values),

                "mean_iou":
                    float(
                        values.mean()
                    ),

                "median_iou":
                    float(
                        np.median(values)
                    ),

                "iou_50":
                    float(
                        (
                            values
                            >= 0.50
                        ).mean()
                    ),

                "iou_25":
                    float(
                        (
                            values
                            >= 0.25
                        ).mean()
                    ),
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "LARGE PATCH-DISJOINT "
        "REFERENCE GROUNDING TRAINING"
    )
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Boundary quantiles: "
        f"{QUANTILE_LOW:.2f}-"
        f"{QUANTILE_HIGH:.2f}"
    )

    # ========================================================
    # CACHE LOAD
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING DOFA CACHE")
    print("=" * 70)

    dofa_cache = DofaShardCache(
        DOFA_ROOT
    )

    print("\n" + "=" * 70)
    print("LOADING TEXT EMBEDDING CACHE")
    print("=" * 70)

    text_cache = GroundingTextCache(
        TEXT_CACHE_DIR
    )

    # ========================================================
    # DATASETS
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    train_dataset = (
        ReferenceGroundingDataset(
            TRAIN_PARQUET,
            dofa_cache,
            text_cache,
        )
    )

    val_dataset = (
        ReferenceGroundingDataset(
            VAL_PARQUET,
            dofa_cache,
            text_cache,
        )
    )

    # ========================================================
    # PATCH ISOLATION
    # ========================================================

    train_patches = set(
        train_dataset.df[
            "patch_id"
        ]
        .astype(str)
    )

    val_patches = set(
        val_dataset.df[
            "patch_id"
        ]
        .astype(str)
    )

    overlap = (
        train_patches
        &
        val_patches
    )

    print("\n" + "=" * 70)
    print("PATCH ISOLATION CHECK")
    print("=" * 70)

    print(
        f"Training patches: "
        f"{len(train_patches):,}"
    )

    print(
        f"Validation patches: "
        f"{len(val_patches):,}"
    )

    print(
        f"Overlap: "
        f"{len(overlap):,}"
    )

    if overlap:

        raise RuntimeError(
            "Patch overlap detected. "
            "Training aborted."
        )

    print(
        "Patch-disjoint validation confirmed."
    )

    # ========================================================
    # ROW COUNTS
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET SIZES")
    print("=" * 70)

    print(
        f"Training rows: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation rows: "
        f"{len(val_dataset):,}"
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_reference_batch,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_reference_batch,
        pin_memory=False,
    )

    print(
        f"\nTraining batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader):,}"
    )

    # ========================================================
    # DATA SMOKE TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("DATA SMOKE TEST")
    print("=" * 70)

    sample_batch = next(
        iter(train_loader)
    )

    print(
        f"S1 shape: "
        f"{tuple(sample_batch['s1'].shape)}"
    )

    print(
        f"S2 shape: "
        f"{tuple(sample_batch['s2'].shape)}"
    )

    print(
        f"Text shape: "
        f"{tuple(sample_batch['text_embedding'].shape)}"
    )

    print(
        f"Target shape: "
        f"{tuple(sample_batch['target_box'].shape)}"
    )

    print(
        f"First patch: "
        f"{sample_batch['patch_id'][0]}"
    )

    print(
        f"First phrase: "
        f"{sample_batch['phrase'][0]}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)

    model = GroundingHeatmapModel()

    model.to(
        DEVICE
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
        )
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_iou = -float("inf")

    history = []

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
        )

        val_metrics = evaluate(
            model,
            val_loader,
        )

        scheduler.step()

        lr = (
            optimizer.param_groups[
                0
            ]["lr"]
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_metrics[
                        "loss"
                    ],

                "heatmap_loss":
                    train_metrics[
                        "heatmap_loss"
                    ],

                "box_loss":
                    train_metrics[
                        "box_loss"
                    ],

                "val_mean_iou":
                    val_metrics[
                        "mean_iou"
                    ],

                "val_median_iou":
                    val_metrics[
                        "median_iou"
                    ],

                "val_iou_50":
                    val_metrics[
                        "iou_50"
                    ],

                "val_iou_25":
                    val_metrics[
                        "iou_25"
                    ],

                "learning_rate":
                    lr,
            }
        )

        print("\n" + "-" * 70)

        print(
            f"Epoch {epoch:02d}/"
            f"{NUM_EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_metrics['loss']:.5f}"
        )

        print(
            f"Heatmap Loss: "
            f"{train_metrics['heatmap_loss']:.5f}"
        )

        print(
            f"Box Loss: "
            f"{train_metrics['box_loss']:.5f}"
        )

        print(
            f"Validation Mean IoU: "
            f"{val_metrics['mean_iou']:.6f}"
        )

        print(
            f"Validation Median IoU: "
            f"{val_metrics['median_iou']:.6f}"
        )

        print(
            f"Validation IoU >= 0.50: "
            f"{100.0 * val_metrics['iou_50']:.2f}%"
        )

        print(
            f"Validation IoU >= 0.25: "
            f"{100.0 * val_metrics['iou_25']:.2f}%"
        )

        print(
            f"Learning rate: "
            f"{lr:.8f}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # Save latest
        # ----------------------------------------------------

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "val_mean_iou":
                    val_metrics[
                        "mean_iou"
                    ],

                "config":
                    {
                        "seed":
                            SEED,

                        "batch_size":
                            BATCH_SIZE,

                        "epochs":
                            NUM_EPOCHS,

                        "learning_rate":
                            LEARNING_RATE,

                        "weight_decay":
                            WEIGHT_DECAY,

                        "quantile_low":
                            QUANTILE_LOW,

                        "quantile_high":
                            QUANTILE_HIGH,
                    },
            },
            LAST_CHECKPOINT,
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if (
            val_metrics[
                "mean_iou"
            ]
            >
            best_iou
        ):

            best_iou = (
                val_metrics[
                    "mean_iou"
                ]
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "val_mean_iou":
                        best_iou,

                    "config":
                        {
                            "seed":
                                SEED,

                            "batch_size":
                                BATCH_SIZE,

                            "epochs":
                                NUM_EPOCHS,

                            "learning_rate":
                                LEARNING_RATE,

                            "weight_decay":
                                WEIGHT_DECAY,

                            "quantile_low":
                                QUANTILE_LOW,

                            "quantile_high":
                                QUANTILE_HIGH,
                        },
                },
                BEST_CHECKPOINT,
            )

            print(
                "\nNEW BEST CHECKPOINT."
            )

            print(
                f"Best validation IoU: "
                f"{best_iou:.6f}"
            )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )

    # ========================================================
    # LOAD BEST MODEL FOR FINAL ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL BEST-CHECKPOINT ANALYSIS")
    print("=" * 70)

    best_checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    best_metrics = evaluate(
        model,
        val_loader,
    )

    phrase_stats = (
        calculate_phrase_statistics(
            best_metrics[
                "records"
            ]
        )
    )

    phrase_stats_path = os.path.join(
        CHECKPOINT_DIR,
        "validation_per_phrase.csv",
    )

    phrase_stats = (
        phrase_stats
        .sort_values(
            "mean_iou"
        )
        .reset_index(
            drop=True
        )
    )

    phrase_stats.to_csv(
        phrase_stats_path,
        index=False,
    )

    # ========================================================
    # CATEGORY-LIKE SUMMARY
    #
    # All examples are reference examples,
    # so we summarize by phrase frequency bins.
    # ========================================================

    print(
        f"\nBest validation Mean IoU: "
        f"{best_metrics['mean_iou']:.6f}"
    )

    print(
        f"Best validation Median IoU: "
        f"{best_metrics['median_iou']:.6f}"
    )

    print(
        f"Validation IoU >= 0.50: "
        f"{100.0 * best_metrics['iou_50']:.2f}%"
    )

    print(
        f"Validation IoU >= 0.25: "
        f"{100.0 * best_metrics['iou_25']:.2f}%"
    )

    # --------------------------------------------------------
    # Worst/best phrases
    # --------------------------------------------------------

    if len(phrase_stats) > 0:

        print(
            "\nWorst 10 validation phrases:"
        )

        for _, row in phrase_stats.head(
            10
        ).iterrows():

            print(
                f"  n={int(row['count']):3d} "
                f"IoU={row['mean_iou']:.4f} "
                f"{row['phrase']}"
            )

        print(
            "\nBest 10 validation phrases:"
        )

        for _, row in phrase_stats.tail(
            10
        ).sort_values(
            "mean_iou",
            ascending=False,
        ).iterrows():

            print(
                f"  n={int(row['count']):3d} "
                f"IoU={row['mean_iou']:.4f} "
                f"{row['phrase']}"
            )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest checkpoint:"
        f"\n{BEST_CHECKPOINT}"
    )

    print(
        f"\nLatest checkpoint:"
        f"\n{LAST_CHECKPOINT}"
    )

    print(
        f"\nTraining history:"
        f"\n{HISTORY_PATH}"
    )

    print(
        f"\nPer-phrase validation statistics:"
        f"\n{phrase_stats_path}"
    )

    print(
        "\nArchitecture unchanged."
    )

    print(
        "Training/validation split is patch-disjoint."
    )

    print(
        f"Final boundary extraction: "
        f"{QUANTILE_LOW:.2f}-"
        f"{QUANTILE_HIGH:.2f}"
    )

    print(
        "\nTEST COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()