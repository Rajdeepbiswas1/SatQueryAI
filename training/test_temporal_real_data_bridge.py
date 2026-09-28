from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import torch


# ============================================================
# Make the SatQueryAI project root importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.temporal_change_vqa import TemporalChangeVQA
# ============================================================
# Paths
# ============================================================

ROOT = PROJECT_ROOT

MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "temporal_manifest.csv"
)

QUESTION_CACHE = (
    ROOT
    / "data"
    / "processed"
    / "temporal_question_embeddings"
)

TEMPORAL_FEATURES = (
    ROOT
    / "data"
    / "processed"
    / "temporal_features"
)


# ============================================================
# Expected architecture / feature dimensions
# ============================================================

EXPECTED_MODEL_PARAMS = 1_744_925

QUESTION_DIM = 768
TOKEN_COUNT = 196
FUSED_DIM = 256

EXPECTED_QUERY_TOKENS = 14


# ============================================================
# Temporal question-type → expected answer-head dimension
# ============================================================

QUESTION_TYPE_TO_DIM = {
    "change_or_not": 2,
    "increase_or_not": 2,
    "decrease_or_not": 2,
    "smallest_change": 6,
    "largest_change": 6,
    "change_to_what": 6,
    "change_ratio": 11,
    "change_ratio_types": 9,
}


# ============================================================
# Utilities
# ============================================================

def normalize_pair_id(value) -> str:
    """
    Normalize temporal pair identifiers to the verified
    five-digit representation used by the DOFA feature cache.

    Examples:
        3       -> 00003
        '3'     -> 00003
        '00003' -> 00003
    """

    text = str(value).strip()

    try:
        return f"{int(float(text)):05d}"
    except ValueError:
        return text


def load_question_embedding(question_id: int) -> torch.Tensor:
    """
    Load exactly one question embedding using the verified
    question-ID → shard mapping.

    The temporal cache was created in ascending question-ID
    order with 4096 rows per shard.
    """

    shard_size = 4096

    shard_idx = question_id // shard_size
    row_idx = question_id % shard_size

    path = QUESTION_CACHE / (
        f"shard_{shard_idx:03d}.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Question-embedding shard missing:\n{path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    ids = data["ids"]
    embeddings = data["question_embeddings"]

    if row_idx >= len(ids):
        raise RuntimeError(
            f"Question ID {question_id} maps to row "
            f"{row_idx}, but {path.name} only contains "
            f"{len(ids)} rows."
        )

    cached_id = int(ids[row_idx])

    if cached_id != question_id:
        raise RuntimeError(
            f"Question-ID mismatch in {path.name}: "
            f"expected {question_id}, found {cached_id}"
        )

    embedding = embeddings[row_idx]

    if tuple(embedding.shape) != (QUESTION_DIM,):
        raise RuntimeError(
            f"Question embedding shape mismatch for "
            f"question_id={question_id}: "
            f"{tuple(embedding.shape)}"
        )

    if embedding.dtype != torch.float32:
        raise RuntimeError(
            f"Question embedding dtype mismatch for "
            f"question_id={question_id}: "
            f"{embedding.dtype}"
        )

    if not torch.isfinite(embedding).all():
        raise RuntimeError(
            f"Non-finite question embedding for "
            f"question_id={question_id}"
        )

    return embedding


def locate_temporal_pairs(pair_ids: set[str]):
    """
    Scan temporal feature shards once and locate the requested
    pair IDs without loading all shards simultaneously.
    """

    shard_paths = sorted(
        TEMPORAL_FEATURES.glob("shard_*.pt")
    )

    if not shard_paths:
        raise FileNotFoundError(
            f"No temporal feature shards found in:\n"
            f"{TEMPORAL_FEATURES}"
        )

    locations = {}

    print()
    print(
        "Locating required temporal pairs across",
        len(shard_paths),
        "feature shards..."
    )

    for path in shard_paths:

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        ids = data["pair_ids"]

        for row_idx, pair_id in enumerate(ids):

            normalized = normalize_pair_id(
                pair_id
            )

            if normalized in pair_ids:
                locations[normalized] = (
                    path,
                    row_idx,
                )

        if len(locations) == len(pair_ids):
            break

    missing = pair_ids - set(locations.keys())

    if missing:
        raise RuntimeError(
            "Required temporal pair IDs were not found "
            f"in the feature cache: {sorted(missing)}"
        )

    return locations


def load_temporal_features(
    locations,
):
    """
    Load only the requested temporal feature rows.
    """

    by_shard = {}

    for pair_id, (path, row_idx) in locations.items():
        by_shard.setdefault(path, []).append(
            (pair_id, row_idx)
        )

    result = {}

    for path, requests in by_shard.items():

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        t1 = data["t1_features"]
        t2 = data["t2_features"]
        diff = data["diff_features"]

        for pair_id, row_idx in requests:

            t1_row = t1[row_idx].float()
            t2_row = t2[row_idx].float()
            diff_row = diff[row_idx].float()

            expected = (
                TOKEN_COUNT,
                QUESTION_DIM,
            )

            if tuple(t1_row.shape) != expected:
                raise RuntimeError(
                    f"T1 shape mismatch for {pair_id}: "
                    f"{tuple(t1_row.shape)}"
                )

            if tuple(t2_row.shape) != expected:
                raise RuntimeError(
                    f"T2 shape mismatch for {pair_id}: "
                    f"{tuple(t2_row.shape)}"
                )

            if tuple(diff_row.shape) != expected:
                raise RuntimeError(
                    f"Difference shape mismatch for {pair_id}: "
                    f"{tuple(diff_row.shape)}"
                )

            if not torch.isfinite(t1_row).all():
                raise RuntimeError(
                    f"Non-finite T1 features for {pair_id}"
                )

            if not torch.isfinite(t2_row).all():
                raise RuntimeError(
                    f"Non-finite T2 features for {pair_id}"
                )

            if not torch.isfinite(diff_row).all():
                raise RuntimeError(
                    f"Non-finite difference features for {pair_id}"
                )

            result[pair_id] = (
                t1_row,
                t2_row,
                diff_row,
            )

    return result


def call_model(
    model,
    t1,
    t2,
    diff,
    question,
):
    """
    Call TemporalChangeVQA using its actual forward signature.

    This avoids making assumptions about argument names while
    still requiring all real temporal inputs.
    """

    signature = inspect.signature(
        model.forward
    )

    parameters = signature.parameters

    values = {}

    aliases = {
        "t1": t1,
        "t1_features": t1,
        "time1": t1,
        "temporal_t1": t1,
        "x_t1": t1,

        "t2": t2,
        "t2_features": t2,
        "time2": t2,
        "temporal_t2": t2,
        "x_t2": t2,

        "diff": diff,
        "diff_features": diff,
        "difference": diff,
        "difference_features": diff,
        "delta": diff,
        "delta_features": diff,

        "question": question,
        "question_embedding": question,
        "question_embeddings": question,
        "query": question,
        "query_embedding": question,
    }

    for name, parameter in parameters.items():

        if name == "self":
            continue

        if name in aliases:
            values[name] = aliases[name]

    required_unmapped = []

    for name, parameter in parameters.items():

        if name == "self":
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
            and name not in values
        ):
            required_unmapped.append(name)

    if required_unmapped:
        raise RuntimeError(
            "Could not map required TemporalChangeVQA "
            f"forward arguments: {required_unmapped}\n"
            f"Actual signature: {signature}"
        )

    return model(**values)


def unpack_outputs(outputs):
    """
    Accept either dictionary-style or tuple/list-style
    TemporalChangeVQA outputs.
    """

    if isinstance(outputs, dict):

        answer_logits = outputs.get(
            "answer_logits"
        )

        query_attention = outputs.get(
            "query_attention"
        )

        change_logits = outputs.get(
            "change_logits"
        )

        fused_tokens = outputs.get(
            "fused_tokens"
        )

        joint = outputs.get(
            "joint"
        )

        if answer_logits is None:
            raise RuntimeError(
                "Model output dictionary does not contain "
                "'answer_logits'."
            )

        return (
            answer_logits,
            query_attention,
            change_logits,
            fused_tokens,
            joint,
        )

    if (
        isinstance(outputs, (tuple, list))
        and len(outputs) >= 5
    ):

        return (
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            outputs[4],
        )

    raise RuntimeError(
        "Unsupported TemporalChangeVQA output format: "
        f"{type(outputs).__name__}"
    )


def get_answer_head_tensors(
    answer_logits,
):
    """
    Normalize the multi-head answer outputs into a list
    of named tensors.

    The real-data bridge identifies heads by their unique
    class dimension:
        2, 6, 9, 11
    """

    if isinstance(answer_logits, dict):

        tensors = {}

        for key, value in answer_logits.items():

            if not torch.is_tensor(value):
                continue

            if value.ndim != 2:
                continue

            tensors[str(key)] = value

        return tensors

    if torch.is_tensor(answer_logits):

        if answer_logits.ndim != 3:
            raise RuntimeError(
                "Answer logits tensor must be 3D when "
                "provided as a single tensor."
            )

        tensors = {}

        for idx in range(
            answer_logits.shape[1]
        ):

            tensor = answer_logits[:, idx, :]

            tensors[
                f"head_{idx}"
            ] = tensor

        return tensors

    if isinstance(answer_logits, (tuple, list)):

        tensors = {}

        for idx, value in enumerate(
            answer_logits
        ):

            if torch.is_tensor(value):
                tensors[
                    f"head_{idx}"
                ] = value

        return tensors

    raise RuntimeError(
        "Unsupported answer_logits type: "
        f"{type(answer_logits).__name__}"
    )


def select_unique_head(
    head_tensors,
    expected_dim,
):
    """
    Select the unique answer head whose class dimension
    matches the expected vocabulary size.
    """

    matches = []

    for name, tensor in head_tensors.items():

        if (
            tensor.ndim == 2
            and tensor.shape[-1] == expected_dim
        ):
            matches.append(
                (name, tensor)
            )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one answer head with "
            f"dimension {expected_dim}, found "
            f"{len(matches)}: "
            f"{[name for name, _ in matches]}"
        )

    return matches[0]


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 72)
    print("SATQUERYAI — TEMPORAL REAL-DATA BRIDGE TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    for path in (
        MANIFEST,
        QUESTION_CACHE,
        TEMPORAL_FEATURES,
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"Required asset not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load verified manifest
    # --------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST
    )

    required_columns = {
        "pair_id",
        "question_id",
        "question_type",
        "question",
        "answer",
    }

    missing = (
        required_columns
        - set(manifest.columns)
    )

    if missing:
        raise RuntimeError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    if len(manifest) != 65_967:
        raise RuntimeError(
            f"Expected 65,967 manifest rows, "
            f"found {len(manifest)}"
        )

    # --------------------------------------------------------
    # Select one deterministic real example from each
    # temporal question type.
    # --------------------------------------------------------

    samples = []

    for question_type in QUESTION_TYPE_TO_DIM:

        subset = manifest[
            manifest["question_type"]
            == question_type
        ]

        if subset.empty:
            raise RuntimeError(
                f"No manifest rows found for "
                f"question type: {question_type}"
            )

        samples.append(
            subset.iloc[0]
        )

    samples = pd.DataFrame(
        samples
    ).reset_index(drop=True)

    print()
    print("Real temporal examples selected:", len(samples))

    for row in samples.itertuples(index=False):

        print(
            f"  {row.question_type:22s}"
            f" qid={int(row.question_id):5d}"
            f" pair={normalize_pair_id(row.pair_id)}"
        )

    # --------------------------------------------------------
    # Load question embeddings
    # --------------------------------------------------------

    question_embeddings = []

    for row in samples.itertuples(index=False):

        qid = int(row.question_id)

        embedding = load_question_embedding(
            qid
        )

        question_embeddings.append(
            embedding
        )

    question_batch = torch.stack(
        question_embeddings,
        dim=0,
    )

    # --------------------------------------------------------
    # Locate temporal image feature pairs
    # --------------------------------------------------------

    pair_ids = {
        normalize_pair_id(row.pair_id)
        for row in samples.itertuples(
            index=False
        )
    }

    locations = locate_temporal_pairs(
        pair_ids
    )

    temporal_features = load_temporal_features(
        locations
    )

    # --------------------------------------------------------
    # Assemble real batch
    # --------------------------------------------------------

    t1_batch = []
    t2_batch = []
    diff_batch = []

    for row in samples.itertuples(index=False):

        pair_id = normalize_pair_id(
            row.pair_id
        )

        t1, t2, diff = temporal_features[
            pair_id
        ]

        t1_batch.append(t1)
        t2_batch.append(t2)
        diff_batch.append(diff)

    t1_batch = torch.stack(
        t1_batch,
        dim=0,
    )

    t2_batch = torch.stack(
        t2_batch,
        dim=0,
    )

    diff_batch = torch.stack(
        diff_batch,
        dim=0,
    )

    # --------------------------------------------------------
    # Validate assembled real inputs
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("REAL INPUT SHAPES")
    print("=" * 72)

    print(
        "T1:",
        tuple(t1_batch.shape),
    )

    print(
        "T2:",
        tuple(t2_batch.shape),
    )

    print(
        "Difference:",
        tuple(diff_batch.shape),
    )

    print(
        "Question:",
        tuple(question_batch.shape),
    )

    expected_batch_size = len(samples)

    if tuple(t1_batch.shape) != (
        expected_batch_size,
        TOKEN_COUNT,
        QUESTION_DIM,
    ):
        raise RuntimeError(
            "T1 batch shape check failed."
        )

    if tuple(t2_batch.shape) != (
        expected_batch_size,
        TOKEN_COUNT,
        QUESTION_DIM,
    ):
        raise RuntimeError(
            "T2 batch shape check failed."
        )

    if tuple(diff_batch.shape) != (
        expected_batch_size,
        TOKEN_COUNT,
        QUESTION_DIM,
    ):
        raise RuntimeError(
            "Difference batch shape check failed."
        )

    if tuple(question_batch.shape) != (
        expected_batch_size,
        QUESTION_DIM,
    ):
        raise RuntimeError(
            "Question batch shape check failed."
        )

    # --------------------------------------------------------
    # Load verified architecture
    # --------------------------------------------------------

    model = TemporalChangeVQA()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        "TemporalChangeVQA parameters:",
        parameter_count,
    )

    if parameter_count != EXPECTED_MODEL_PARAMS:
        raise RuntimeError(
            f"Parameter-count mismatch: "
            f"expected {EXPECTED_MODEL_PARAMS}, "
            f"found {parameter_count}"
        )

    model.eval()

    # --------------------------------------------------------
    # Real-data inference
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("REAL-DATA FORWARD PASS")
    print("=" * 72)

    with torch.inference_mode():

        outputs = call_model(
            model,
            t1_batch,
            t2_batch,
            diff_batch,
            question_batch,
        )

    (
        answer_logits,
        query_attention,
        change_logits,
        fused_tokens,
        joint,
    ) = unpack_outputs(outputs)

    # --------------------------------------------------------
    # Auxiliary output validation
    # --------------------------------------------------------

    print(
        "Query attention:",
        None
        if query_attention is None
        else tuple(query_attention.shape),
    )

    print(
        "Change logits:",
        None
        if change_logits is None
        else tuple(change_logits.shape),
    )

    print(
        "Fused tokens:",
        None
        if fused_tokens is None
        else tuple(fused_tokens.shape),
    )

    print(
        "Joint:",
        None
        if joint is None
        else tuple(joint.shape),
    )

    if query_attention is None:
        raise RuntimeError(
            "query_attention was not returned."
        )

    if change_logits is None:
        raise RuntimeError(
            "change_logits was not returned."
        )

    if fused_tokens is None:
        raise RuntimeError(
            "fused_tokens was not returned."
        )

    if joint is None:
        raise RuntimeError(
            "joint was not returned."
        )

    if tuple(
        query_attention.shape
    ) != (
        expected_batch_size,
        EXPECTED_QUERY_TOKENS,
        EXPECTED_QUERY_TOKENS,
    ):
        raise RuntimeError(
            "Query-attention shape mismatch."
        )

    if tuple(
        change_logits.shape
    ) != (
        expected_batch_size,
        EXPECTED_QUERY_TOKENS,
        EXPECTED_QUERY_TOKENS,
    ):
        raise RuntimeError(
            "Change-logit shape mismatch."
        )

    if tuple(
        fused_tokens.shape
    ) != (
        expected_batch_size,
        TOKEN_COUNT,
        FUSED_DIM,
    ):
        raise RuntimeError(
            "Fused-token shape mismatch."
        )

    if tuple(
        joint.shape
    ) != (
        expected_batch_size,
        FUSED_DIM,
    ):
        raise RuntimeError(
            "Joint representation shape mismatch."
        )

    # --------------------------------------------------------
    # Finite-value checks
    # --------------------------------------------------------

    tensors_to_check = {
        "answer_logits":
            answer_logits,
        "query_attention":
            query_attention,
        "change_logits":
            change_logits,
        "fused_tokens":
            fused_tokens,
        "joint":
            joint,
    }

    for name, value in tensors_to_check.items():

        if torch.is_tensor(value):

            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"Non-finite values in {name}."
                )

        elif isinstance(value, dict):

            for key, tensor in value.items():

                if not torch.is_tensor(tensor):
                    raise RuntimeError(
                        f"{name}[{key}] is not a tensor."
                    )

                if not torch.isfinite(tensor).all():
                    raise RuntimeError(
                        f"Non-finite values in "
                        f"{name}[{key}]."
                    )

    print()
    print("Finite-value checks: PASS")

    # --------------------------------------------------------
    # Inspect answer heads
    # --------------------------------------------------------

    head_tensors = get_answer_head_tensors(
        answer_logits
    )

    if not head_tensors:
        raise RuntimeError(
            "No answer-head tensors could be extracted."
        )

    print()
    print("=" * 72)
    print("ANSWER HEADS")
    print("=" * 72)

    for name, tensor in head_tensors.items():

        print(
            f"{name:24s} "
            f"{tuple(tensor.shape)}"
        )

    # --------------------------------------------------------
    # Real question-type routing validation
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("REAL QUESTION-TYPE ROUTING")
    print("=" * 72)

    for row in samples.itertuples(index=False):

        question_type = str(
            row.question_type
        ).strip()

        expected_dim = QUESTION_TYPE_TO_DIM[
            question_type
        ]

        head_name, head = select_unique_head(
            head_tensors,
            expected_dim,
        )

        print(
            f"{question_type:22s}"
            f" -> {head_name}"
            f" -> {tuple(head.shape)}"
            f" -> expected {expected_dim}"
        )

    # --------------------------------------------------------
    # Attention normalization check
    #
    # The model applies softmax over all 196 spatial tokens
    # before reshaping them into a [14,14] attention map.
    #
    # Therefore the COMPLETE spatial map must sum to 1 for
    # each batch element. Individual rows are not expected
    # to sum to 1.
    # --------------------------------------------------------

    attention_sum = (
        query_attention
        .sum(dim=-1)
        .sum(dim=-1)
    )

    attention_error = (
        attention_sum - 1.0
    ).abs().max().item()

    print()
    print(
        "Maximum full-map attention normalization error:",
        attention_error,
    )

    if attention_error > 1e-4:
        raise RuntimeError(
            "Full-map query-attention normalization check failed."
        )

    print(
        "Full-map attention normalization: PASS"
    )
    print(
        "Attention normalization: PASS"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TEMPORAL REAL-DATA BRIDGE VERIFIED")
    print("=" * 72)

    print(
        "Real QA examples:",
        len(samples),
    )

    print(
        "Real temporal question embeddings:",
        len(question_embeddings),
    )

    print(
        "Real temporal image pairs:",
        len(pair_ids),
    )

    print(
        "Question IDs:",
        "verified through cache",
    )

    print(
        "Pair IDs:",
        "verified through feature shards",
    )

    print(
        "Model parameter count:",
        parameter_count,
    )

    print(
        "Temporal model forward:",
        "PASS",
    )

    print(
        "Answer-head routing:",
        "PASS",
    )

    print()
    print("=" * 72)
    print("TEMPORAL REAL-DATA BRIDGE TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()