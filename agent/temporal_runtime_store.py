# =============================================================================
# SATQUERYAI — TEMPORAL RUNTIME STORE
# =============================================================================
#
# Runtime data bridge for the verified temporal Test cache.
#
# Responsibilities:
#   - Load the held-out temporal Test manifest.
#   - Resolve question_id -> pair_id.
#   - Resolve pair_id -> feature shard + row.
#   - Load temporal feature shards lazily.
#   - Load temporal question embeddings lazily.
#   - Return the exact tensors required by TemporalChangeV3Tool.
#
# This file does NOT:
#   - modify model architecture
#   - train anything
#   - modify checkpoints
#   - modify authoritative caches
#
# =============================================================================

from __future__ import annotations

import csv
import gc
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(r"D:\SatQueryAI")

TEST_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_split"
    / "test_manifest.csv"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features"
)

QUESTION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_test_question_embeddings"
)

RUNTIME_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_runtime"
)

PAIR_INDEX_PATH = (
    RUNTIME_ROOT
    / "temporal_pair_index.json"
)


# =============================================================================
# INVARIANTS
# =============================================================================

EXPECTED_QUESTION_ROWS = 39_686
EXPECTED_TEST_PAIRS = 968
EXPECTED_FEATURE_PAIRS = 2_968

EXPECTED_FEATURE_SHAPE = (196, 768)

SUPPORTED_QUESTION_TYPES = {
    "change_or_not",
    "increase_or_not",
    "decrease_or_not",
    "smallest_change",
    "largest_change",
    "change_to_what",
    "change_ratio",
    "change_ratio_types",
}


# =============================================================================
# BASIC VALIDATION
# =============================================================================

def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def _load_pt(path: Path) -> dict:
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected dictionary in {path}, got {type(data)}"
        )

    return data


# =============================================================================
# PAIR-ID NORMALIZATION
# =============================================================================

def _normalize_pair_id(value: Any) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(
            f"Could not normalize pair ID: {value!r}"
        ) from exc


# =============================================================================
# BUILD PAIR INDEX
# =============================================================================

def build_temporal_pair_index(
    feature_root: Path = FEATURE_ROOT,
    output_path: Path = PAIR_INDEX_PATH,
) -> Dict[str, Any]:
    """
    Scan the authoritative temporal feature shards once and create a tiny
    pair_id -> {shard, row} runtime index.

    The feature tensors themselves are not retained after each shard is read.
    """

    _require_path(
        feature_root,
        "Temporal feature directory",
    )

    files = sorted(
        feature_root.glob("shard_*.pt")
    )

    if not files:
        files = sorted(
            feature_root.glob("*.pt")
        )

    if not files:
        raise FileNotFoundError(
            f"No temporal feature shards found under {feature_root}"
        )

    pair_index: Dict[str, Dict[str, Any]] = {}

    total_rows = 0

    print("=" * 80)
    print("SATQUERYAI — BUILDING TEMPORAL PAIR INDEX")
    print("=" * 80)

    for shard_number, path in enumerate(files, start=1):

        print()
        print(
            f"[{shard_number}/{len(files)}] Loading "
            f"{path.name}"
        )

        data = _load_pt(path)

        pair_ids = data.get("pair_ids")
        t1 = data.get("t1_features")
        t2 = data.get("t2_features")
        diff = data.get("diff_features")

        if pair_ids is None:
            raise RuntimeError(
                f"{path}: missing pair_ids"
            )

        if t1 is None or t2 is None or diff is None:
            raise RuntimeError(
                f"{path}: missing temporal feature tensors"
            )

        if not (
            t1.shape == t2.shape == diff.shape
        ):
            raise RuntimeError(
                f"{path}: temporal tensor shape mismatch: "
                f"{tuple(t1.shape)}, "
                f"{tuple(t2.shape)}, "
                f"{tuple(diff.shape)}"
            )

        if t1.ndim != 3:
            raise RuntimeError(
                f"{path}: expected 3D tensors, "
                f"got {tuple(t1.shape)}"
            )

        if tuple(t1.shape[1:]) != EXPECTED_FEATURE_SHAPE:
            raise RuntimeError(
                f"{path}: invalid feature shape "
                f"{tuple(t1.shape)}; expected "
                f"[N,196,768]"
            )

        if len(pair_ids) != t1.shape[0]:
            raise RuntimeError(
                f"{path}: pair count {len(pair_ids)} "
                f"does not match tensor rows {t1.shape[0]}"
            )

        for row_index, raw_pair_id in enumerate(pair_ids):

            pair_id = _normalize_pair_id(
                raw_pair_id
            )

            if pair_id in pair_index:
                raise RuntimeError(
                    f"Duplicate temporal pair ID detected: "
                    f"{pair_id}"
                )

            pair_index[pair_id] = {
                "shard": path.name,
                "row": int(row_index),
            }

        total_rows += int(t1.shape[0])

        print(
            f"  pairs: {t1.shape[0]}"
        )

        del data
        del pair_ids
        del t1
        del t2
        del diff

        gc.collect()

    if total_rows != EXPECTED_FEATURE_PAIRS:
        raise RuntimeError(
            "Unexpected total temporal feature pair count: "
            f"expected {EXPECTED_FEATURE_PAIRS}, "
            f"got {total_rows}"
        )

    if len(pair_index) != EXPECTED_FEATURE_PAIRS:
        raise RuntimeError(
            "Unexpected unique temporal pair count: "
            f"expected {EXPECTED_FEATURE_PAIRS}, "
            f"got {len(pair_index)}"
        )

    # -------------------------------------------------------------------------
    # Atomic JSON write
    # -------------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": 1,
        "feature_root": str(feature_root),
        "feature_shape": list(EXPECTED_FEATURE_SHAPE),
        "total_pairs": len(pair_index),
        "pairs": {
            str(pair_id): location
            for pair_id, location
            in pair_index.items()
        },
    }

    temporary_path = output_path.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(
        output_path
    )

    print()
    print(
        f"Pair index written to: {output_path}"
    )
    print(
        f"Indexed pairs: {len(pair_index)}"
    )

    return payload


# =============================================================================
# TEMPORAL RUNTIME STORE
# =============================================================================

class TemporalRuntimeStore:
    """
    Lazy runtime bridge for temporal Test data.

    The large feature shards are loaded one at a time and cached using a small
    LRU cache. The question embedding shards are handled the same way.
    """

    def __init__(
        self,
        manifest_path: Path = TEST_MANIFEST_PATH,
        feature_root: Path = FEATURE_ROOT,
        question_root: Path = QUESTION_ROOT,
        pair_index_path: Path = PAIR_INDEX_PATH,
        max_feature_shards: int = 1,
        max_question_shards: int = 1,
    ) -> None:

        self.manifest_path = Path(
            manifest_path
        )

        self.feature_root = Path(
            feature_root
        )

        self.question_root = Path(
            question_root
        )

        self.pair_index_path = Path(
            pair_index_path
        )

        self.max_feature_shards = max(
            1,
            int(max_feature_shards),
        )

        self.max_question_shards = max(
            1,
            int(max_question_shards),
        )

        _require_path(
            self.manifest_path,
            "Temporal Test manifest",
        )

        _require_path(
            self.feature_root,
            "Temporal feature directory",
        )

        _require_path(
            self.question_root,
            "Temporal Test question embedding directory",
        )

        if not self.pair_index_path.exists():
            build_temporal_pair_index(
                feature_root=self.feature_root,
                output_path=self.pair_index_path,
            )

        self.manifest = self._load_manifest()

        self.pair_index = self._load_pair_index()

        self.question_locations = (
            self._build_question_locations()
        )

        self._feature_cache = OrderedDict()
        self._question_cache = OrderedDict()

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------

    def _load_manifest(
        self,
    ) -> Dict[int, Dict[str, Any]]:

        rows: Dict[int, Dict[str, Any]] = {}

        with self.manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:

            reader = csv.DictReader(handle)

            required_columns = {
                "pair_id",
                "file_name",
                "question_id",
                "question_type",
                "question",
                "answer",
            }

            if not reader.fieldnames:
                raise RuntimeError(
                    "Temporal Test manifest has no header."
                )

            missing = (
                required_columns
                - set(reader.fieldnames)
            )

            if missing:
                raise RuntimeError(
                    "Temporal Test manifest missing columns: "
                    f"{sorted(missing)}"
                )

            for row in reader:

                question_id = int(
                    row["question_id"]
                )

                if question_id in rows:
                    raise RuntimeError(
                        f"Duplicate question_id in manifest: "
                        f"{question_id}"
                    )

                question_type = (
                    row["question_type"]
                )

                if (
                    question_type
                    not in SUPPORTED_QUESTION_TYPES
                ):
                    raise RuntimeError(
                        f"Unsupported temporal question type "
                        f"{question_type!r} at question_id="
                        f"{question_id}"
                    )

                rows[question_id] = {
                    "pair_id": _normalize_pair_id(
                        row["pair_id"]
                    ),
                    "file_name": row["file_name"],
                    "question_id": question_id,
                    "question_type": question_type,
                    "question": row["question"],
                    "answer": row["answer"],
                }

        if len(rows) != EXPECTED_QUESTION_ROWS:
            raise RuntimeError(
                "Unexpected temporal Test manifest size: "
                f"expected {EXPECTED_QUESTION_ROWS}, "
                f"got {len(rows)}"
            )

        expected_ids = set(
            range(EXPECTED_QUESTION_ROWS)
        )

        if set(rows) != expected_ids:
            raise RuntimeError(
                "Temporal Test question IDs are not exactly "
                "0..39685."
            )

        test_pairs = {
            row["pair_id"]
            for row in rows.values()
        }

        if len(test_pairs) != EXPECTED_TEST_PAIRS:
            raise RuntimeError(
                "Unexpected temporal Test pair count: "
                f"expected {EXPECTED_TEST_PAIRS}, "
                f"got {len(test_pairs)}"
            )

        return rows

    # -------------------------------------------------------------------------
    # Pair index
    # -------------------------------------------------------------------------

    def _load_pair_index(
        self,
    ) -> Dict[int, Dict[str, Any]]:

        with self.pair_index_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            payload = json.load(handle)

        if payload.get("total_pairs") != (
            EXPECTED_FEATURE_PAIRS
        ):
            raise RuntimeError(
                "Temporal pair index has unexpected size."
            )

        raw_pairs = payload.get("pairs")

        if not isinstance(
            raw_pairs,
            dict,
        ):
            raise RuntimeError(
                "Temporal pair index is missing 'pairs'."
            )

        result = {
            int(pair_id): {
                "shard": str(location["shard"]),
                "row": int(location["row"]),
            }
            for pair_id, location
            in raw_pairs.items()
        }

        if len(result) != EXPECTED_FEATURE_PAIRS:
            raise RuntimeError(
                "Temporal pair index contains unexpected "
                "number of unique pair IDs."
            )

        return result

    # -------------------------------------------------------------------------
    # Question shard indexing
    # -------------------------------------------------------------------------

    def _build_question_locations(
        self,
    ) -> Dict[int, Tuple[str, int]]:

        files = sorted(
            self.question_root.glob("shard_*.pt")
        )

        if not files:
            files = sorted(
                self.question_root.glob("*.pt")
            )

        if not files:
            raise FileNotFoundError(
                "No temporal Test question embedding shards found."
            )

        locations: Dict[int, Tuple[str, int]] = {}

        for path in files:

            data = _load_pt(path)

            question_ids = data.get(
                "question_ids"
            )

            embeddings = data.get(
                "embeddings"
            )

            if question_ids is None:
                raise RuntimeError(
                    f"{path}: missing question_ids"
                )

            if embeddings is None:
                raise RuntimeError(
                    f"{path}: missing embeddings"
                )

            if embeddings.ndim != 2:
                raise RuntimeError(
                    f"{path}: invalid embedding shape "
                    f"{tuple(embeddings.shape)}"
                )

            if embeddings.shape[1] != 768:
                raise RuntimeError(
                    f"{path}: expected 768-dimensional "
                    f"embeddings, got {embeddings.shape[1]}"
                )

            if len(question_ids) != embeddings.shape[0]:
                raise RuntimeError(
                    f"{path}: question ID count does not "
                    f"match embedding rows."
                )

            for row_index, raw_question_id in enumerate(
                question_ids
            ):

                question_id = int(
                    raw_question_id
                )

                if question_id in locations:
                    raise RuntimeError(
                        f"Duplicate question ID detected: "
                        f"{question_id}"
                    )

                locations[question_id] = (
                    path.name,
                    int(row_index),
                )

            del data
            del question_ids
            del embeddings

            gc.collect()

        if len(locations) != EXPECTED_QUESTION_ROWS:
            raise RuntimeError(
                "Unexpected number of indexed question embeddings: "
                f"expected {EXPECTED_QUESTION_ROWS}, "
                f"got {len(locations)}"
            )

        return locations

    # -------------------------------------------------------------------------
    # Lazy shard loading
    # -------------------------------------------------------------------------

    def _get_feature_shard(
        self,
        shard_name: str,
    ) -> dict:

        if shard_name in self._feature_cache:

            data = self._feature_cache.pop(
                shard_name
            )

            self._feature_cache[shard_name] = data

            return data

        path = self.feature_root / shard_name

        data = _load_pt(path)

        self._feature_cache[shard_name] = data

        while len(self._feature_cache) > (
            self.max_feature_shards
        ):

            _, evicted = (
                self._feature_cache.popitem(
                    last=False
                )
            )

            del evicted
            gc.collect()

        return data

    def _get_question_shard(
        self,
        shard_name: str,
    ) -> dict:

        if shard_name in self._question_cache:

            data = self._question_cache.pop(
                shard_name
            )

            self._question_cache[shard_name] = data

            return data

        path = self.question_root / shard_name

        data = _load_pt(path)

        self._question_cache[shard_name] = data

        while len(self._question_cache) > (
            self.max_question_shards
        ):

            _, evicted = (
                self._question_cache.popitem(
                    last=False
                )
            )

            del evicted
            gc.collect()

        return data

    # -------------------------------------------------------------------------
    # Public lookup
    # -------------------------------------------------------------------------

    def get(
        self,
        question_id: int,
    ) -> Dict[str, Any]:

        question_id = int(
            question_id
        )

        if question_id not in self.manifest:
            raise KeyError(
                f"Temporal Test question_id not found: "
                f"{question_id}"
            )

        metadata = self.manifest[
            question_id
        ]

        pair_id = metadata["pair_id"]

        if pair_id not in self.pair_index:
            raise KeyError(
                f"Temporal feature pair not found: "
                f"pair_id={pair_id}"
            )

        # ---------------------------------------------------------------------
        # Temporal features
        # ---------------------------------------------------------------------

        feature_location = self.pair_index[
            pair_id
        ]

        feature_data = self._get_feature_shard(
            feature_location["shard"]
        )

        row_index = feature_location["row"]

        t1 = feature_data[
            "t1_features"
        ][row_index]

        t2 = feature_data[
            "t2_features"
        ][row_index]

        diff = feature_data[
            "diff_features"
        ][row_index]

        if tuple(t1.shape) != EXPECTED_FEATURE_SHAPE:
            raise RuntimeError(
                f"Invalid t1 feature shape for pair "
                f"{pair_id}: {tuple(t1.shape)}"
            )

        if tuple(t2.shape) != EXPECTED_FEATURE_SHAPE:
            raise RuntimeError(
                f"Invalid t2 feature shape for pair "
                f"{pair_id}: {tuple(t2.shape)}"
            )

        if tuple(diff.shape) != EXPECTED_FEATURE_SHAPE:
            raise RuntimeError(
                f"Invalid diff feature shape for pair "
                f"{pair_id}: {tuple(diff.shape)}"
            )

        # ---------------------------------------------------------------------
        # Question embedding
        # ---------------------------------------------------------------------

        question_shard_name, question_row = (
            self.question_locations[
                question_id
            ]
        )

        question_data = self._get_question_shard(
            question_shard_name
        )

        question_embedding = (
            question_data["embeddings"][
                question_row
            ]
        )

        if tuple(question_embedding.shape) != (
            768,
        ):
            raise RuntimeError(
                "Invalid temporal question embedding shape: "
                f"{tuple(question_embedding.shape)}"
            )

        # ---------------------------------------------------------------------
        # Return exact TemporalChangeV3Tool inputs.
        # ---------------------------------------------------------------------

        return {
            "question_id": question_id,
            "pair_id": pair_id,
            "file_name": metadata["file_name"],
            "question_type": metadata["question_type"],
            "question": metadata["question"],
            "answer": metadata["answer"],

            "t1": t1.unsqueeze(0).float().contiguous(),
            "t2": t2.unsqueeze(0).float().contiguous(),
            "diff": diff.unsqueeze(0).float().contiguous(),
            "text_embedding": (
                question_embedding
                .unsqueeze(0)
                .float()
                .contiguous()
            ),
        }


# =============================================================================
# DIRECT SELF-TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("SATQUERYAI — TEMPORAL RUNTIME STORE TEST")
    print("=" * 80)

    store = TemporalRuntimeStore()

    sample = store.get(
        question_id=0
    )

    print()
    print("Question ID:", sample["question_id"])
    print("Pair ID:", sample["pair_id"])
    print("File:", sample["file_name"])
    print("Question type:", sample["question_type"])
    print("Question:", sample["question"])
    print("Expected answer:", sample["answer"])

    print()
    print("t1 shape:", tuple(sample["t1"].shape))
    print("t2 shape:", tuple(sample["t2"].shape))
    print("diff shape:", tuple(sample["diff"].shape))
    print(
        "text embedding shape:",
        tuple(sample["text_embedding"].shape),
    )

    if tuple(sample["t1"].shape) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            "t1 runtime shape verification failed."
        )

    if tuple(sample["t2"].shape) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            "t2 runtime shape verification failed."
        )

    if tuple(sample["diff"].shape) != (
        1,
        196,
        768,
    ):
        raise RuntimeError(
            "diff runtime shape verification failed."
        )

    if tuple(sample["text_embedding"].shape) != (
        1,
        768,
    ):
        raise RuntimeError(
            "Question embedding runtime shape "
            "verification failed."
        )

    print()
    print(
        "Temporal runtime store initialization: PASS"
    )
    print(
        "Temporal question lookup: PASS"
    )
    print(
        "Temporal feature lookup: PASS"
    )
    print(
        "Runtime tensor contract: PASS"
    )