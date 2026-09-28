from __future__ import annotations

import inspect
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.temporal_change_vqa import TemporalChangeVQA


# ============================================================
# PATHS
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
# EXPECTED VERIFIED TEMPORAL VOCABULARIES
# ============================================================

BINARY_VOCAB = (
    "no",
    "yes",
)

LAND_COVER_VOCAB = (
    "buildings",
    "trees",
    "low_vegetation",
    "NVG_surface",
    "water",
    "playgrounds",
)

RATIO_VOCAB = (
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

RATIO_TYPE_VOCAB = (
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


QUESTION_TYPE_TO_VOCAB = {
    "change_or_not": BINARY_VOCAB,
    "increase_or_not": BINARY_VOCAB,
    "decrease_or_not": BINARY_VOCAB,
    "smallest_change": LAND_COVER_VOCAB,
    "largest_change": LAND_COVER_VOCAB,
    "change_to_what": LAND_COVER_VOCAB,
    "change_ratio": RATIO_VOCAB,
    "change_ratio_types": RATIO_TYPE_VOCAB,
}


QUESTION_TYPE_TO_HEAD = {
    "change_or_not": "binary",
    "increase_or_not": "binary",
    "decrease_or_not": "binary",
    "smallest_change": "land_cover",
    "largest_change": "land_cover",
    "change_to_what": "land_cover",
    "change_ratio": "change_ratio",
    "change_ratio_types": "change_ratio_types",
}


EXPECTED_HEAD_DIMS = {
    "binary": 2,
    "land_cover": 6,
    "change_ratio": 11,
    "change_ratio_types": 9,
}


EXPECTED_MODEL_PARAMS = 1_744_925


# ============================================================
# FEATURE CACHE HELPERS
# ============================================================

def normalize_pair_id(value) -> str:
    text = str(value).strip()

    try:
        return f"{int(float(text)):05d}"
    except ValueError:
        return text


def load_pair_map():
    """
    Build pair_id -> (shard_path, row_idx).

    Only metadata/index tensors are inspected first.
    """

    print()
    print("=" * 72)
    print("BUILDING TEMPORAL PAIR INDEX")
    print("=" * 72)

    shard_paths = sorted(
        TEMPORAL_FEATURES.glob("shard_*.pt")
    )

    if not shard_paths:
        raise FileNotFoundError(
            f"No temporal feature shards found:\n"
            f"{TEMPORAL_FEATURES}"
        )

    pair_map = {}

    for shard_path in shard_paths:

        data = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        pair_ids = data["pair_ids"]

        for row_idx, pair_id in enumerate(pair_ids):

            normalized = normalize_pair_id(
                pair_id
            )

            if normalized in pair_map:
                raise RuntimeError(
                    f"Duplicate temporal pair ID found: "
                    f"{normalized}"
                )

            pair_map[normalized] = (
                shard_path,
                row_idx,
            )

    print(
        "Feature shards:",
        len(shard_paths),
    )

    print(
        "Indexed temporal pairs:",
        len(pair_map),
    )

    if len(pair_map) != 2_968:
        raise RuntimeError(
            f"Expected 2,968 temporal pairs, "
            f"found {len(pair_map)}"
        )

    return pair_map


def load_pair_features(
    pair_map,
    pair_id,
):
    """
    Load one real temporal feature pair.
    """

    if pair_id not in pair_map:
        raise RuntimeError(
            f"Temporal pair not found: {pair_id}"
        )

    path, row_idx = pair_map[pair_id]

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    t1 = data["t1_features"][row_idx].float()
    t2 = data["t2_features"][row_idx].float()
    diff = data["diff_features"][row_idx].float()

    expected_shape = (
        196,
        768,
    )

    for name, tensor in (
        ("t1", t1),
        ("t2", t2),
        ("diff", diff),
    ):

        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch for "
                f"{pair_id}: {tuple(tensor.shape)}"
            )

        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Non-finite {name} features for "
                f"{pair_id}"
            )

    return t1, t2, diff


# ============================================================
# QUESTION EMBEDDING HELPER
# ============================================================

def load_question_embedding(
    question_id: int,
):
    shard_size = 4096

    shard_idx = (
        question_id
        // shard_size
    )

    row_idx = (
        question_id
        % shard_size
    )

    path = (
        QUESTION_CACHE
        / f"shard_{shard_idx:03d}.pt"
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

    embeddings = (
        data["question_embeddings"]
    )

    if row_idx >= len(ids):
        raise RuntimeError(
            f"Question ID {question_id} "
            f"maps beyond shard bounds."
        )

    cached_id = int(
        ids[row_idx]
    )

    if cached_id != question_id:
        raise RuntimeError(
            f"Question-ID mismatch: "
            f"expected {question_id}, "
            f"found {cached_id}"
        )

    embedding = embeddings[row_idx]

    if tuple(embedding.shape) != (
        768,
    ):
        raise RuntimeError(
            f"Question embedding shape mismatch "
            f"for {question_id}: "
            f"{tuple(embedding.shape)}"
        )

    if embedding.dtype != torch.float32:
        raise RuntimeError(
            f"Question embedding dtype mismatch "
            f"for {question_id}: "
            f"{embedding.dtype}"
        )

    if not torch.isfinite(embedding).all():
        raise RuntimeError(
            f"Non-finite question embedding "
            f"for {question_id}"
        )

    return embedding


# ============================================================
# MODEL CALL
# ============================================================

def call_temporal_model(
    model,
    t1,
    t2,
    diff,
    question,
):
    """
    Call the model using its actual forward signature.
    """

    signature = inspect.signature(
        model.forward
    )

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

    values = {}

    for name, parameter in (
        signature.parameters.items()
    ):

        if name == "self":
            continue

        if name in aliases:
            values[name] = aliases[name]

    missing = []

    for name, parameter in (
        signature.parameters.items()
    ):

        if name == "self":
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
            and name not in values
        ):
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Unable to map required model "
            f"arguments: {missing}\n"
            f"Actual signature: {signature}"
        )

    return model(**values)


# ============================================================
# OUTPUT EXTRACTION
# ============================================================

def extract_answer_logits(outputs):
    if not isinstance(outputs, dict):
        raise RuntimeError(
            "Expected TemporalChangeVQA to return "
            f"a dictionary, found {type(outputs).__name__}"
        )

    if "answer_logits" not in outputs:
        raise RuntimeError(
            "Model output is missing "
            "'answer_logits'."
        )

    answer_logits = outputs[
        "answer_logits"
    ]

    if not isinstance(
        answer_logits,
        dict,
    ):
        raise RuntimeError(
            "Expected answer_logits to be a "
            "dictionary of named heads."
        )

    return answer_logits


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)
    print("SATQUERYAI — TEMPORAL LABEL / LOSS BRIDGE TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # Path checks
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
    # Load manifest
    # --------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST
    )

    expected_rows = 65_967

    if len(manifest) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows:,} "
            f"manifest rows, found {len(manifest):,}"
        )

    required_columns = {
        "pair_id",
        "question_id",
        "question_type",
        "question",
        "answer",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing manifest columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Verify question types
    # --------------------------------------------------------

    observed_question_types = set(
        manifest[
            "question_type"
        ]
        .astype(str)
        .str.strip()
    )

    expected_question_types = set(
        QUESTION_TYPE_TO_VOCAB.keys()
    )

    print()
    print(
        "Observed question types:",
        sorted(observed_question_types),
    )

    if (
        observed_question_types
        != expected_question_types
    ):
        raise RuntimeError(
            "Temporal question-type vocabulary mismatch."
        )

    print(
        "Question-type vocabulary: PASS"
    )

    # --------------------------------------------------------
    # Verify every answer maps to the expected vocabulary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FULL MANIFEST LABEL VERIFICATION")
    print("=" * 72)

    total_label_rows = 0

    type_counts = Counter()

    answer_counts = {
        question_type: Counter()
        for question_type
        in QUESTION_TYPE_TO_VOCAB
    }

    target_index_counts = {
        question_type: Counter()
        for question_type
        in QUESTION_TYPE_TO_VOCAB
    }

    for row in manifest.itertuples(
        index=False
    ):

        question_type = str(
            row.question_type
        ).strip()

        answer = str(
            row.answer
        ).strip()

        vocabulary = (
            QUESTION_TYPE_TO_VOCAB[
                question_type
            ]
        )

        if answer not in vocabulary:

            raise RuntimeError(
                "Unexpected answer label:\n"
                f"  question_type = {question_type}\n"
                f"  question_id   = {int(row.question_id)}\n"
                f"  answer        = {answer}\n"
                f"  valid labels  = {vocabulary}"
            )

        target_index = vocabulary.index(
            answer
        )

        if not (
            0
            <= target_index
            < len(vocabulary)
        ):
            raise RuntimeError(
                f"Target index out of range for "
                f"{question_type}: "
                f"{target_index}"
            )

        type_counts[
            question_type
        ] += 1

        answer_counts[
            question_type
        ][answer] += 1

        target_index_counts[
            question_type
        ][target_index] += 1

        total_label_rows += 1

    if total_label_rows != expected_rows:
        raise RuntimeError(
            f"Processed {total_label_rows:,} labels "
            f"but expected {expected_rows:,}"
        )

    for question_type, vocabulary in (
        QUESTION_TYPE_TO_VOCAB.items()
    ):

        observed_answers = set(
            answer_counts[
                question_type
            ].keys()
        )

        expected_answers = set(
            vocabulary
        )

        if observed_answers != expected_answers:
            raise RuntimeError(
                "Observed answer vocabulary mismatch "
                f"for {question_type}.\n"
                f"Observed: {sorted(observed_answers)}\n"
                f"Expected: {sorted(expected_answers)}"
            )

    print(
        "Manifest label rows:",
        total_label_rows,
    )

    print(
        "All answers belong to expected vocabularies: PASS"
    )

    print(
        "All target indices within range: PASS"
    )

    # --------------------------------------------------------
    # Print complete mapping tables
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("CLASS INDEX MAPPINGS")
    print("=" * 72)

    for question_type, vocabulary in (
        QUESTION_TYPE_TO_VOCAB.items()
    ):

        head = (
            QUESTION_TYPE_TO_HEAD[
                question_type
            ]
        )

        print()
        print(
            question_type,
            "->",
            head,
        )

        for index, label in enumerate(
            vocabulary
        ):

            count = answer_counts[
                question_type
            ][label]

            print(
                f"  {index:2d} -> "
                f"{label:16s} "
                f"n={count:,}"
            )

    # --------------------------------------------------------
    # Verify model architecture
    # --------------------------------------------------------

    model = TemporalChangeVQA()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("=" * 72)
    print("MODEL / HEAD VERIFICATION")
    print("=" * 72)

    print(
        "Model parameters:",
        parameter_count,
    )

    if parameter_count != EXPECTED_MODEL_PARAMS:
        raise RuntimeError(
            f"Expected {EXPECTED_MODEL_PARAMS:,} "
            f"model parameters, found "
            f"{parameter_count:,}"
        )

    print(
        "Parameter count: PASS"
    )

    model.eval()

    # --------------------------------------------------------
    # Select one real example from EVERY question type.
    #
    # This gives a real batch covering all eight routes.
    # --------------------------------------------------------

    sample_rows = []

    for question_type in (
        QUESTION_TYPE_TO_VOCAB.keys()
    ):

        subset = manifest[
            manifest["question_type"]
            == question_type
        ]

        if subset.empty:
            raise RuntimeError(
                f"No examples for {question_type}"
            )

        sample_rows.append(
            subset.iloc[0]
        )

    samples = pd.DataFrame(
        sample_rows
    ).reset_index(drop=True)

    print()
    print(
        "Real samples selected:",
        len(samples),
    )

    # --------------------------------------------------------
    # Build real inputs
    # --------------------------------------------------------

    pair_map = load_pair_map()

    t1_batch = []
    t2_batch = []
    diff_batch = []
    question_batch = []

    target_indices = []
    selected_types = []
    selected_answers = []

    for row in samples.itertuples(
        index=False
    ):

        question_type = str(
            row.question_type
        ).strip()

        answer = str(
            row.answer
        ).strip()

        qid = int(
            row.question_id
        )

        pair_id = normalize_pair_id(
            row.pair_id
        )

        vocabulary = (
            QUESTION_TYPE_TO_VOCAB[
                question_type
            ]
        )

        target_index = vocabulary.index(
            answer
        )

        t1, t2, diff = (
            load_pair_features(
                pair_map,
                pair_id,
            )
        )

        question_embedding = (
            load_question_embedding(
                qid
            )
        )

        t1_batch.append(t1)
        t2_batch.append(t2)
        diff_batch.append(diff)
        question_batch.append(
            question_embedding
        )

        target_indices.append(
            target_index
        )

        selected_types.append(
            question_type
        )

        selected_answers.append(
            answer
        )

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

    question_batch = torch.stack(
        question_batch,
        dim=0,
    )

    print()
    print("=" * 72)
    print("REAL LOSS-BRIDGE INPUTS")
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

    expected_batch = 8

    if tuple(t1_batch.shape) != (
        expected_batch,
        196,
        768,
    ):
        raise RuntimeError(
            "T1 shape check failed."
        )

    if tuple(t2_batch.shape) != (
        expected_batch,
        196,
        768,
    ):
        raise RuntimeError(
            "T2 shape check failed."
        )

    if tuple(diff_batch.shape) != (
        expected_batch,
        196,
        768,
    ):
        raise RuntimeError(
            "Difference shape check failed."
        )

    if tuple(question_batch.shape) != (
        expected_batch,
        768,
    ):
        raise RuntimeError(
            "Question shape check failed."
        )

    print(
        "Real input shapes: PASS"
    )

    # --------------------------------------------------------
    # Real model forward
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("REAL MODEL FORWARD")
    print("=" * 72)

    with torch.inference_mode():

        outputs = call_temporal_model(
            model,
            t1_batch,
            t2_batch,
            diff_batch,
            question_batch,
        )

    answer_logits = extract_answer_logits(
        outputs
    )

    print(
        "Available answer heads:",
        list(answer_logits.keys()),
    )

    # --------------------------------------------------------
    # Verify exact model head dimensions
    # --------------------------------------------------------

    for head_name, expected_dim in (
        EXPECTED_HEAD_DIMS.items()
    ):

        if head_name not in answer_logits:
            raise RuntimeError(
                f"Required model head missing: "
                f"{head_name}"
            )

        logits = answer_logits[
            head_name
        ]

        if not torch.is_tensor(logits):
            raise RuntimeError(
                f"Head '{head_name}' "
                "is not a tensor."
            )

        expected_shape = (
            expected_batch,
            expected_dim,
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"Head '{head_name}' shape mismatch: "
                f"{tuple(logits.shape)} "
                f"!= {expected_shape}"
            )

        if not torch.isfinite(
            logits
        ).all():
            raise RuntimeError(
                f"Non-finite logits in "
                f"{head_name}"
            )

    print(
        "All model head dimensions: PASS"
    )

    # --------------------------------------------------------
    # Verify each question type selects its intended head
    # and its target index is legal for that head.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("HEAD / TARGET / LOSS VERIFICATION")
    print("=" * 72)

    per_type_losses = []

    for batch_index in range(
        expected_batch
    ):

        question_type = (
            selected_types[
                batch_index
            ]
        )

        answer = (
            selected_answers[
                batch_index
            ]
        )

        head_name = (
            QUESTION_TYPE_TO_HEAD[
                question_type
            ]
        )

        vocabulary = (
            QUESTION_TYPE_TO_VOCAB[
                question_type
            ]
        )

        target_index = (
            target_indices[
                batch_index
            ]
        )

        logits = answer_logits[
            head_name
        ]

        expected_dim = len(
            vocabulary
        )

        if logits.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Head dimension mismatch for "
                f"{question_type}: "
                f"{logits.shape[-1]} "
                f"!= {expected_dim}"
            )

        if not (
            0
            <= target_index
            < logits.shape[-1]
        ):
            raise RuntimeError(
                f"Illegal target index for "
                f"{question_type}: "
                f"{target_index}"
            )

        # ----------------------------------------------------
        # One real sample → intended head → CrossEntropyLoss
        # ----------------------------------------------------

        sample_logits = logits[
            batch_index:
            batch_index + 1
        ]

        target = torch.tensor(
            [target_index],
            dtype=torch.long,
        )

        loss = F.cross_entropy(
            sample_logits,
            target,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss for "
                f"{question_type}"
            )

        per_type_losses.append(
            loss
        )

        print(
            f"{question_type:22s}"
            f" answer={answer:16s}"
            f" -> {head_name:18s}"
            f" -> class={target_index:2d}"
            f" -> CE={loss.item():.6f}"
        )

    print()
    print(
        "All head/target mappings: PASS"
    )

    print(
        "All per-sample CrossEntropy losses: PASS"
    )

    # --------------------------------------------------------
    # Verify a combined finite loss can be formed without
    # changing the model. This is a forward-only loss test;
    # no backward pass is performed here.
    # --------------------------------------------------------

    combined_loss = torch.stack(
        per_type_losses
    ).mean()

    print()
    print(
        "Combined forward-only loss:",
        combined_loss.item(),
    )

    if not torch.isfinite(
        combined_loss
    ):
        raise RuntimeError(
            "Combined loss is non-finite."
        )

    print(
        "Combined loss finiteness: PASS"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TEMPORAL LABEL / LOSS BRIDGE VERIFIED")
    print("=" * 72)

    print(
        "Manifest labels checked:",
        total_label_rows,
    )

    print(
        "Question types checked:",
        len(QUESTION_TYPE_TO_VOCAB),
    )

    print(
        "Real examples checked:",
        expected_batch,
    )

    print(
        "Model parameters:",
        parameter_count,
    )

    print(
        "Label vocabulary:",
        "PASS",
    )

    print(
        "Target index mapping:",
        "PASS",
    )

    print(
        "Correct head selection:",
        "PASS",
    )

    print(
        "CrossEntropy loss:",
        "PASS",
    )

    print()
    print("=" * 72)
    print("TEMPORAL LABEL / LOSS BRIDGE TEST PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()