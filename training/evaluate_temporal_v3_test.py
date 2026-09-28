from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.temporal_change_vqa_v2 import (
    TemporalChangeVQAv2,
    count_trainable_parameters,
)


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(r"D:\SatQueryAI")

TEST_MANIFEST = (
    ROOT / "data" / "processed" / "temporal_split" / "test_manifest.csv"
)

FEATURE_ROOT = (
    ROOT / "data" / "processed" / "temporal_features"
)

TEST_Q_ROOT = (
    ROOT / "data" / "processed"
    / "temporal_test_question_embeddings"
)

CHECKPOINT = (
    ROOT / "models" / "checkpoints"
    / "temporal_change_vqa_v3_ordinal"
    / "best.pt"
)

OUTPUT_ROOT = (
    ROOT / "outputs" / "temporal_change_vqa_v3_test"
)

BATCH_SIZE = 32
EXPECTED_PARAMS = 1_679_133
EXPECTED_TEST_PAIRS = 968
EXPECTED_TEST_ROWS = 39_686


# =============================================================================
# EXACT TRAINING VOCABULARIES
# =============================================================================

BINARY = (
    "no",
    "yes",
)

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

VOCAB = {
    "binary": BINARY,
    "land_cover": LAND_COVER,
    "change_ratio": RATIO,
    "change_ratio_types": RATIO_TYPE,
}

HEAD_TO_ID = {
    head: {label: i for i, label in enumerate(labels)}
    for head, labels in VOCAB.items()
}


# =============================================================================
# HELPERS
# =============================================================================

def normalize_pair_id(value) -> str:
    text = str(value).strip()

    if text.isdigit():
        return text.zfill(5)

    return text


def load_feature_store(root: Path):
    files = sorted(root.glob("shard_*.pt"))

    if not files:
        raise RuntimeError(
            f"No feature shards found under {root}"
        )

    pair_ids = []
    t1_parts = []
    t2_parts = []
    diff_parts = []

    print()
    print("=" * 80)
    print("LOADING TEMPORAL DOFA FEATURES")
    print("=" * 80)

    for index, path in enumerate(files, start=1):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        pair_ids_part = data.get("pair_ids")

        t1 = data.get("t1")
        if t1 is None:
            t1 = data.get("t1_features")

        t2 = data.get("t2")
        if t2 is None:
            t2 = data.get("t2_features")

        diff = data.get("diff")
        if diff is None:
            diff = data.get("diff_features")

        if (
            pair_ids_part is None
            or t1 is None
            or t2 is None
            or diff is None
        ):
            raise RuntimeError(
                f"{path.name}: expected pair_ids + t1/t2/diff"
            )

        if not (
            t1.shape == t2.shape == diff.shape
        ):
            raise RuntimeError(
                f"{path.name}: feature shape mismatch"
            )

        if (
            t1.ndim != 3
            or tuple(t1.shape[1:]) != (196, 768)
        ):
            raise RuntimeError(
                f"{path.name}: invalid feature shape "
                f"{tuple(t1.shape)}"
            )

        pair_ids.extend(
            normalize_pair_id(x)
            for x in pair_ids_part
        )

        t1_parts.append(
            t1.contiguous()
        )

        t2_parts.append(
            t2.contiguous()
        )

        diff_parts.append(
            diff.contiguous()
        )

        print(
            f"Shard {index:02d}/{len(files):02d} | "
            f"pairs={t1.shape[0]}"
        )

    t1_all = torch.cat(
        t1_parts,
        dim=0,
    )

    t2_all = torch.cat(
        t2_parts,
        dim=0,
    )

    diff_all = torch.cat(
        diff_parts,
        dim=0,
    )

    if len(pair_ids) != 2968:
        raise RuntimeError(
            f"Expected 2968 feature pairs, found {len(pair_ids)}"
        )

    lookup = {
        pair_id: i
        for i, pair_id in enumerate(pair_ids)
    }

    if len(lookup) != len(pair_ids):
        raise RuntimeError(
            "Duplicate temporal pair IDs detected."
        )

    print(
        f"Feature pairs: {len(pair_ids):,}"
    )

    print(
        f"T1 shape: {tuple(t1_all.shape)}"
    )

    print(
        f"T2 shape: {tuple(t2_all.shape)}"
    )

    print(
        f"Diff shape: {tuple(diff_all.shape)}"
    )

    return pair_ids, lookup, t1_all, t2_all, diff_all


def load_question_embeddings(root: Path, expected_rows: int):
    files = sorted(root.glob("shard_*.pt"))

    if not files:
        raise RuntimeError(
            f"No Test question embedding shards found under {root}"
        )

    ids = []
    parts = []

    print()
    print("=" * 80)
    print("LOADING TEST QUESTION EMBEDDINGS")
    print("=" * 80)

    for index, path in enumerate(files, start=1):

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        qids = data["question_ids"]
        embeddings = data["embeddings"]

        ids.extend(
            int(x)
            for x in qids.tolist()
        )

        parts.append(
            embeddings.float().contiguous()
        )

        print(
            f"Shard {index:02d}/{len(files):02d} | "
            f"rows={embeddings.shape[0]}"
        )

    embeddings_all = torch.cat(
        parts,
        dim=0,
    )

    if len(ids) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} question embeddings, "
            f"found {len(ids)}"
        )

    if ids != list(range(expected_rows)):
        raise RuntimeError(
            "Test question IDs are not exactly 0..N-1."
        )

    if tuple(embeddings_all.shape) != (
        expected_rows,
        768,
    ):
        raise RuntimeError(
            f"Unexpected Test embedding shape: "
            f"{tuple(embeddings_all.shape)}"
        )

    if embeddings_all.dtype != torch.float32:
        raise RuntimeError(
            "Test embeddings are not float32."
        )

    if not torch.isfinite(
        embeddings_all
    ).all():
        raise RuntimeError(
            "Non-finite Test question embeddings detected."
        )

    print(
        f"Question embeddings: {tuple(embeddings_all.shape)}"
    )

    return embeddings_all


class TemporalTestDataset(Dataset):

    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_lookup,
        t1,
        t2,
        diff,
        question_embeddings,
    ):
        self.df = manifest.reset_index(drop=True)
        self.feature_lookup = feature_lookup
        self.t1 = t1
        self.t2 = t2
        self.diff = diff
        self.question_embeddings = question_embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        pair_id = normalize_pair_id(
            row["pair_id"]
        )

        feature_index = self.feature_lookup[
            pair_id
        ]

        question_id = int(
            row["question_id"]
        )

        return {
            "pair_id": pair_id,
            "question_id": question_id,
            "qtype": str(row["question_type"]).strip(),
            "answer": str(row["answer"]).strip(),
            "t1": self.t1[feature_index],
            "t2": self.t2[feature_index],
            "diff": self.diff[feature_index],
            "question": self.question_embeddings[
                question_id
            ],
        }


def collate(batch):
    return {
        "pair_id": [x["pair_id"] for x in batch],
        "question_id": torch.tensor(
            [x["question_id"] for x in batch],
            dtype=torch.long,
        ),
        "qtype": [x["qtype"] for x in batch],
        "answer": [x["answer"] for x in batch],
        "t1": torch.stack(
            [x["t1"] for x in batch]
        ),
        "t2": torch.stack(
            [x["t2"] for x in batch]
        ),
        "diff": torch.stack(
            [x["diff"] for x in batch]
        ),
        "question": torch.stack(
            [x["question"] for x in batch]
        ),
    }


def update_stats(stats, predictions, qtypes, answers):

    for prediction, qtype, answer in zip(
        predictions,
        qtypes,
        answers,
    ):

        head = QTYPE_TO_HEAD[qtype]

        labels = VOCAB[head]
        label_index = HEAD_TO_ID[head]

        if answer not in label_index:
            raise RuntimeError(
                f"Unexpected answer {answer!r} "
                f"for qtype {qtype!r}"
            )

        true_index = label_index[answer]

        pred_index = int(prediction)

        key = (head, true_index)

        stats[head]["total"] += 1

        if pred_index == true_index:
            stats[head]["correct"] += 1

        stats["qtype"][qtype]["total"] += 1

        if pred_index == true_index:
            stats["qtype"][qtype]["correct"] += 1

        stats["confusion"][head][
            true_index,
            pred_index,
        ] += 1


def make_stats():

    stats = {
        "binary": {
            "correct": 0,
            "total": 0,
        },
        "land_cover": {
            "correct": 0,
            "total": 0,
        },
        "change_ratio": {
            "correct": 0,
            "total": 0,
        },
        "change_ratio_types": {
            "correct": 0,
            "total": 0,
        },
        "qtype": defaultdict(
            lambda: {
                "correct": 0,
                "total": 0,
            }
        ),
        "confusion": {
            head: np.zeros(
                (len(labels), len(labels)),
                dtype=np.int64,
            )
            for head, labels in VOCAB.items()
        },
    }

    return stats


def balanced_accuracy(matrix: np.ndarray) -> float:

    recalls = []

    for i in range(matrix.shape[0]):

        support = int(
            matrix[i].sum()
        )

        if support == 0:
            continue

        recalls.append(
            matrix[i, i] / support
        )

    if not recalls:
        return float("nan")

    return float(
        np.mean(recalls)
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("SATQUERYAI — V3 TEMPORAL HELD-OUT TEST EVALUATION")
    print("=" * 80)

    for path, label in (
        (TEST_MANIFEST, "Test manifest"),
        (FEATURE_ROOT, "Temporal features"),
        (TEST_Q_ROOT, "Test question embeddings"),
        (CHECKPOINT, "V3 checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}"
            )

    # -------------------------------------------------------------------------
    # Test manifest
    # -------------------------------------------------------------------------

    print()
    print("Loading held-out Test manifest...")

    test_df = pd.read_csv(
        TEST_MANIFEST
    )

    required = {
        "pair_id",
        "question_id",
        "question_type",
        "question",
        "answer",
    }

    missing = (
        required
        - set(test_df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Test manifest missing columns: {sorted(missing)}"
        )

    if len(test_df) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_ROWS} Test rows, "
            f"found {len(test_df)}"
        )

    unique_pairs = {
        normalize_pair_id(x)
        for x in test_df["pair_id"]
    }

    if len(unique_pairs) != EXPECTED_TEST_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_PAIRS} Test pairs, "
            f"found {len(unique_pairs)}"
        )

    question_ids = (
        test_df["question_id"]
        .astype(int)
        .tolist()
    )

    if question_ids != list(
        range(EXPECTED_TEST_ROWS)
    ):
        raise RuntimeError(
            "Test question IDs do not exactly match "
            "the established 0..39685 ordering."
        )

    print(
        f"Test rows:  {len(test_df):,}"
    )

    print(
        f"Test pairs: {len(unique_pairs):,}"
    )

    print(
        "Test manifest integrity: PASS"
    )

    # -------------------------------------------------------------------------
    # Load features / embeddings
    # -------------------------------------------------------------------------

    (
        feature_pair_ids,
        feature_lookup,
        t1,
        t2,
        diff,
    ) = load_feature_store(
        FEATURE_ROOT
    )

    question_embeddings = (
        load_question_embeddings(
            TEST_Q_ROOT,
            EXPECTED_TEST_ROWS,
        )
    )

    missing_features = (
        unique_pairs
        - set(feature_lookup.keys())
    )

    if missing_features:
        raise RuntimeError(
            f"Missing Test feature pairs: "
            f"{sorted(missing_features)[:10]}"
        )

    print(
        "Test pair → feature mapping: PASS"
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    dataset = TemporalTestDataset(
        manifest=test_df,
        feature_lookup=feature_lookup,
        t1=t1,
        t2=t2,
        diff=diff,
        question_embeddings=question_embeddings,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    print(
        f"Test batches: {len(loader):,}"
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING V3 CHECKPOINT")
    print("=" * 80)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    model = TemporalChangeVQAv2(
        input_dim=768,
        hidden_dim=256,
        num_spatial_tokens=196,
        grid_size=14,
    ).to(device)

    parameter_count = count_trainable_parameters(
        model
    )

    print(
        "Parameter count:",
        parameter_count,
    )

    if parameter_count != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Parameter count mismatch: "
            f"expected {EXPECTED_PARAMS}, "
            f"got {parameter_count}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    print(
        "Checkpoint model name:",
        checkpoint.get("model_name"),
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch"),
    )

    print(
        "Checkpoint validation loss:",
        checkpoint.get("val_loss"),
    )

    print(
        "Checkpoint validation accuracy:",
        checkpoint.get("val_accuracy"),
    )

    print(
        "Checkpoint validation balanced accuracy:",
        checkpoint.get("val_balanced_accuracy"),
    )

    recorded_params = checkpoint.get(
        "parameter_count"
    )

    if recorded_params != EXPECTED_PARAMS:
        raise RuntimeError(
            "Checkpoint parameter_count metadata mismatch."
        )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:
        raise RuntimeError(
            "Checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        "Strict checkpoint loading: PASS"
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    stats = make_stats()

    total_start = time.time()

    print()
    print("=" * 80)
    print("RUNNING HELD-OUT TEST INFERENCE")
    print("=" * 80)

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            batch_t1 = batch["t1"].to(
                device
            )

            batch_t2 = batch["t2"].to(
                device
            )

            batch_diff = batch["diff"].to(
                device
            )

            batch_question = batch["question"].to(
                device
            )

            outputs = model(
                t1=batch_t1,
                t2=batch_t2,
                diff=batch_diff,
                question=batch_question,
            )

            answer_logits = outputs[
                "answer_logits"
            ]

            # One prediction is selected from the head corresponding
            # to each question type.
            predictions = []

            for qtype in batch["qtype"]:

                head = QTYPE_TO_HEAD[qtype]

                logits = answer_logits[
                    head
                ]

                # We handle the whole batch below by selecting rows.
                predictions.append(head)

            # Group indices by head.
            head_indices = defaultdict(list)

            for i, qtype in enumerate(
                batch["qtype"]
            ):
                head = QTYPE_TO_HEAD[qtype]
                head_indices[head].append(i)

            final_predictions = [
                None
                for _ in batch["qtype"]
            ]

            for head, indices in head_indices.items():

                index_tensor = torch.tensor(
                    indices,
                    dtype=torch.long,
                    device=device,
                )

                head_logits = answer_logits[
                    head
                ].index_select(
                    0,
                    index_tensor,
                )

                head_predictions = (
                    head_logits.argmax(dim=-1)
                    .detach()
                    .cpu()
                    .tolist()
                )

                for local_index, original_index in enumerate(
                    indices
                ):
                    final_predictions[
                        original_index
                    ] = head_predictions[
                        local_index
                    ]

            update_stats(
                stats,
                final_predictions,
                batch["qtype"],
                batch["answer"],
            )

            if (
                batch_index == 1
                or batch_index % 50 == 0
                or batch_index == len(loader)
            ):
                processed = min(
                    batch_index * BATCH_SIZE,
                    EXPECTED_TEST_ROWS,
                )

                elapsed = (
                    time.time()
                    - total_start
                )

                print(
                    f"Batch {batch_index:4d}/{len(loader):4d} | "
                    f"rows {processed:6d}/{EXPECTED_TEST_ROWS:6d} | "
                    f"elapsed {elapsed:8.1f}s"
                )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    total_correct = sum(
        stats[head]["correct"]
        for head in VOCAB
    )

    total_count = sum(
        stats[head]["total"]
        for head in VOCAB
    )

    overall_accuracy = (
        total_correct / total_count
        if total_count
        else float("nan")
    )

    head_metrics = {}

    for head in VOCAB:

        correct = stats[head]["correct"]
        total = stats[head]["total"]

        matrix = stats["confusion"][
            head
        ]

        head_metrics[head] = {
            "accuracy": (
                correct / total
                if total
                else float("nan")
            ),
            "balanced_accuracy": balanced_accuracy(
                matrix
            ),
            "samples": total,
        }

    valid_balanced = [
        x["balanced_accuracy"]
        for x in head_metrics.values()
        if not math.isnan(
            x["balanced_accuracy"]
        )
    ]

    overall_balanced_accuracy = (
        float(np.mean(valid_balanced))
        if valid_balanced
        else float("nan")
    )

    qtype_metrics = {}

    for qtype, values in sorted(
        stats["qtype"].items()
    ):

        qtype_metrics[qtype] = {
            "accuracy": (
                values["correct"]
                / values["total"]
                if values["total"]
                else float("nan")
            ),
            "correct": values["correct"],
            "samples": values["total"],
        }

    print()
    print("=" * 80)
    print("V3 HELD-OUT TEST RESULTS")
    print("=" * 80)

    print(
        f"Overall accuracy:          {overall_accuracy:.6f}"
    )

    print(
        f"Overall balanced accuracy: {overall_balanced_accuracy:.6f}"
    )

    print()
    print("Metrics by head:")

    for head, values in head_metrics.items():

        print(
            f"  {head:22s} "
            f"accuracy={values['accuracy']:.6f} "
            f"balanced={values['balanced_accuracy']:.6f} "
            f"samples={values['samples']}"
        )

    print()
    print("Metrics by question type:")

    for qtype, values in qtype_metrics.items():

        print(
            f"  {qtype:22s} "
            f"accuracy={values['accuracy']:.6f} "
            f"samples={values['samples']}"
        )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "model": "TemporalChangeVQAv3OrdinalLoss",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "checkpoint_val_accuracy": checkpoint.get(
            "val_accuracy"
        ),
        "checkpoint_val_balanced_accuracy": checkpoint.get(
            "val_balanced_accuracy"
        ),
        "parameter_count": parameter_count,
        "test_rows": len(test_df),
        "test_pairs": len(unique_pairs),
        "overall_accuracy": overall_accuracy,
        "overall_balanced_accuracy": overall_balanced_accuracy,
        "head_metrics": head_metrics,
        "question_type_metrics": qtype_metrics,
        "device": str(device),
    }

    (OUTPUT_ROOT / "v3_test_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    for head, matrix in stats[
        "confusion"
    ].items():

        labels = list(
            VOCAB[head]
        )

        confusion_df = pd.DataFrame(
            matrix,
            index=[
                f"true_{x}"
                for x in labels
            ],
            columns=[
                f"pred_{x}"
                for x in labels
            ],
        )

        confusion_df.to_csv(
            OUTPUT_ROOT
            / f"confusion_{head}.csv"
        )

    print()
    print(
        "Results written to:",
        OUTPUT_ROOT,
    )

    print()
    print("=" * 80)
    print(
        "SATQUERYAI V3 HELD-OUT TEST EVALUATION: COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()