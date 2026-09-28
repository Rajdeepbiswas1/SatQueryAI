from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")
FEATURE_DIR = ROOT / "data" / "processed" / "temporal_features"
TRAIN_Q_DIR = ROOT / "data" / "processed" / "temporal_question_embeddings"
VAL_Q_DIR = ROOT / "data" / "processed" / "temporal_val_question_embeddings"
TRAIN_MANIFEST = ROOT / "data" / "processed" / "temporal_split" / "train_manifest.csv"
VAL_MANIFEST = ROOT / "data" / "processed" / "temporal_split" / "val_manifest.csv"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.temporal_change_vqa import TemporalChangeVQA


# ============================================================
# HELPERS
# ============================================================

def tensor_mb(x: torch.Tensor) -> float:
    return x.numel() * x.element_size() / (1024 ** 2)


def load_feature_shards():
    shard_paths = sorted(FEATURE_DIR.glob("*.pt"))

    if len(shard_paths) != 12:
        raise RuntimeError(
            f"Expected exactly 12 temporal feature shards, found {len(shard_paths)}"
        )

    store = {}

    print(f"[1/6] Loading {len(shard_paths)} temporal feature shards...")

    total_bytes = 0

    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        obj = torch.load(shard_path, map_location="cpu")

        required = {"pair_ids", "t1_features", "t2_features", "diff_features"}
        missing = required - set(obj.keys())

        if missing:
            raise RuntimeError(
                f"{shard_path.name}: missing keys {sorted(missing)}"
            )

        pair_ids = obj["pair_ids"]
        t1 = obj["t1_features"]
        t2 = obj["t2_features"]
        diff = obj["diff_features"]

        if not (
            len(pair_ids) == len(t1) == len(t2) == len(diff)
        ):
            raise RuntimeError(
                f"{shard_path.name}: pair/tensor row counts do not match"
            )

        if t1.ndim != 3 or t2.ndim != 3 or diff.ndim != 3:
            raise RuntimeError(
                f"{shard_path.name}: expected [N,196,768] tensors"
            )

        if t1.shape[1:] != (196, 768):
            raise RuntimeError(
                f"{shard_path.name}: unexpected t1 shape {tuple(t1.shape)}"
            )

        if t2.shape[1:] != (196, 768):
            raise RuntimeError(
                f"{shard_path.name}: unexpected t2 shape {tuple(t2.shape)}"
            )

        if diff.shape[1:] != (196, 768):
            raise RuntimeError(
                f"{shard_path.name}: unexpected diff shape {tuple(diff.shape)}"
            )

        for row_idx, pair_id in enumerate(pair_ids):
            pid = int(pair_id)

            if pid in store:
                raise RuntimeError(
                    f"Duplicate pair_id detected: {pid}"
                )

            store[pid] = {
                "t1": t1[row_idx],
                "t2": t2[row_idx],
                "diff": diff[row_idx],
            }

        total_bytes += (
            t1.numel() * t1.element_size()
            + t2.numel() * t2.element_size()
            + diff.numel() * diff.element_size()
        )

        print(
            f"    shard {shard_idx:02d}: "
            f"{len(pair_ids):4d} pairs | "
            f"{tensor_mb(t1) + tensor_mb(t2) + tensor_mb(diff):8.2f} MB"
        )

        del obj

    print(f"    total unique pairs loaded: {len(store)}")
    print(f"    raw feature tensor size: {total_bytes / (1024 ** 3):.3f} GiB")

    if len(store) != 2968:
        raise RuntimeError(
            f"Expected 2968 unique pairs, found {len(store)}"
        )

    print("    PASS: all 2968 temporal pairs loaded.")
    return store


def load_question_cache(cache_dir: Path, expected_rows: int):
    shard_paths = sorted(cache_dir.glob("*.pt"))

    if not shard_paths:
        raise RuntimeError(f"No question-cache shards found in {cache_dir}")

    cache = {}

    print(f"    loading {len(shard_paths)} question shards from:")
    print(f"    {cache_dir}")

    for shard_path in shard_paths:
        obj = torch.load(shard_path, map_location="cpu")

        if {"ids", "question_embeddings"}.issubset(obj.keys()):
            question_ids = obj["ids"]
            embeddings = obj["question_embeddings"]

        elif {"question_ids", "embeddings"}.issubset(obj.keys()):
            question_ids = obj["question_ids"]
            embeddings = obj["embeddings"]

        else:
            raise RuntimeError(
                f"{shard_path.name}: unsupported question-cache schema. "
                f"Found keys: {list(obj.keys())}"
            )

        if embeddings.ndim != 2 or embeddings.shape[1] != 768:
            raise RuntimeError(
                f"{shard_path.name}: unexpected embedding shape "
                f"{tuple(embeddings.shape)}"
            )

        if len(question_ids) != len(embeddings):
            raise RuntimeError(
                f"{shard_path.name}: question IDs and embeddings differ in length"
            )

        for row_idx, qid in enumerate(question_ids):
            qid = int(qid)

            if qid in cache:
                raise RuntimeError(
                    f"Duplicate question ID detected: {qid}"
                )

            cache[qid] = embeddings[row_idx]

    if len(cache) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} cached questions, found {len(cache)}"
        )

    print(
        f"    PASS: {len(cache)} question embeddings loaded, "
        f"shape [N,768]."
    )

    return cache


def main():
    print("=" * 70)
    print("SatQueryAI Temporal RAM Loader Smoke Test")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 1. Temporal features
    # --------------------------------------------------------

    feature_store = load_feature_shards()
    print()

    # --------------------------------------------------------
    # 2. Manifests
    # --------------------------------------------------------

    print("[2/6] Checking train/validation manifests...")

    train_df = pd.read_csv(TRAIN_MANIFEST)
    val_df = pd.read_csv(VAL_MANIFEST)

    print(f"    train rows: {len(train_df)}")
    print(f"    val rows:   {len(val_df)}")

    if len(train_df) != 65967:
        raise RuntimeError(
            f"Expected 65,967 train QA rows, found {len(train_df)}"
        )

    if len(val_df) != 16441:
        raise RuntimeError(
            f"Expected 16,441 validation QA rows, found {len(val_df)}"
        )

    if train_df["pair_id"].nunique() != 1600:
        raise RuntimeError(
            f"Expected 1,600 train pairs, found {train_df['pair_id'].nunique()}"
        )

    if val_df["pair_id"].nunique() != 400:
        raise RuntimeError(
            f"Expected 400 validation pairs, found {val_df['pair_id'].nunique()}"
        )

    print("    PASS: manifest counts match the verified split.")
    print()

    # --------------------------------------------------------
    # 3. Question caches
    # --------------------------------------------------------

    print("[3/6] Loading train/validation question embeddings...")

    train_questions = load_question_cache(
        TRAIN_Q_DIR,
        expected_rows=65967,
    )

    val_questions = load_question_cache(
        VAL_Q_DIR,
        expected_rows=16441,
    )

    print()

    # --------------------------------------------------------
    # 4. Resolve real examples
    # --------------------------------------------------------

    print("[4/6] Resolving real train/validation examples...")

    sample_train = train_df.iloc[0]
    sample_val = val_df.iloc[0]

    train_pair_id = int(sample_train["pair_id"])
    val_pair_id = int(sample_val["pair_id"])

    train_question_id = int(sample_train["question_id"])
    val_question_id = int(sample_val["question_id"])

    for name, pair_id in [
        ("train", train_pair_id),
        ("val", val_pair_id),
    ]:
        if pair_id not in feature_store:
            raise RuntimeError(
                f"{name}: pair_id {pair_id} missing from feature store"
            )

    if train_question_id not in train_questions:
        raise RuntimeError(
            f"Train question_id {train_question_id} missing from cache"
        )

    if val_question_id not in val_questions:
        raise RuntimeError(
            f"Validation question_id {val_question_id} missing from cache"
        )

    train_feature = feature_store[train_pair_id]
    val_feature = feature_store[val_pair_id]

    print(
        f"    train pair {train_pair_id}: "
        f"t1={tuple(train_feature['t1'].shape)}, "
        f"t2={tuple(train_feature['t2'].shape)}, "
        f"diff={tuple(train_feature['diff'].shape)}"
    )

    print(
        f"    val pair {val_pair_id}: "
        f"t1={tuple(val_feature['t1'].shape)}, "
        f"t2={tuple(val_feature['t2'].shape)}, "
        f"diff={tuple(val_feature['diff'].shape)}"
    )

    print(
        f"    train question embedding: "
        f"{tuple(train_questions[train_question_id].shape)}"
    )

    print(
        f"    val question embedding: "
        f"{tuple(val_questions[val_question_id].shape)}"
    )

    print("    PASS: real manifest IDs resolve into both caches.")
    print()

    # --------------------------------------------------------
    # 5. Frozen model interface
    # --------------------------------------------------------

    print("[5/6] Testing the frozen TemporalChangeVQA interface...")

    model = TemporalChangeVQA()

    expected_params = 1_744_925
    actual_params = sum(p.numel() for p in model.parameters())

    print(f"    parameters: {actual_params:,}")

    if actual_params != expected_params:
        raise RuntimeError(
            f"Expected {expected_params:,} parameters, "
            f"found {actual_params:,}"
        )

    model.eval()

    t1 = train_feature["t1"].unsqueeze(0).float()
    t2 = train_feature["t2"].unsqueeze(0).float()
    diff = train_feature["diff"].unsqueeze(0).float()

    question = train_questions[train_question_id].unsqueeze(0).float()

    with torch.no_grad():
        outputs = model(t1, t2, diff, question)

    required_top = {
        "answer_logits",
        "query_attention",
        "change_logits",
        "fused_tokens",
        "joint",
    }

    missing_top = required_top - set(outputs.keys())

    if missing_top:
        raise RuntimeError(
            f"Missing model outputs: {sorted(missing_top)}"
        )

    required_heads = {
        "binary": (1, 2),
        "change_ratio": (1, 11),
        "change_ratio_types": (1, 9),
        "land_cover": (1, 6),
    }

    answer_logits = outputs["answer_logits"]

    if not isinstance(answer_logits, dict):
        raise RuntimeError(
            "outputs['answer_logits'] is not a dictionary"
        )

    for head_name, expected_shape in required_heads.items():
        if head_name not in answer_logits:
            raise RuntimeError(
                f"Missing answer head: {head_name}"
            )

        actual_shape = tuple(answer_logits[head_name].shape)

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{head_name}: expected {expected_shape}, "
                f"found {actual_shape}"
            )

    expected_core = {
        "query_attention": (1, 14, 14),
        "change_logits": (1, 14, 14),
        "fused_tokens": (1, 196, 256),
        "joint": (1, 256),
    }

    for name, expected_shape in expected_core.items():
        actual_shape = tuple(outputs[name].shape)

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{name}: expected {expected_shape}, "
                f"found {actual_shape}"
            )

    print("    PASS: frozen model interface is exact.")
    print()

    # --------------------------------------------------------
    # 6. Cleanup
    # --------------------------------------------------------

    print("[6/6] Cleaning temporary Python objects...")

    del model
    del outputs
    del t1, t2, diff, question
    del train_questions, val_questions
    del train_df, val_df
    del feature_store

    gc.collect()

    print("    PASS")
    print()
    print("=" * 70)
    print("RESULT: TEMPORAL RAM LOADER SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()