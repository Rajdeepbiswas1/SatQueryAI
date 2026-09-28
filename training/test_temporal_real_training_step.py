from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(
    r"D:\SatQueryAI"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ================================================================
# IMPORTS
# ================================================================

from src.data.temporal_dataset import (
    TemporalDOFADataset,
)

from src.models.temporal_change_vqa import (
    TemporalChangeVQA,
)


# ================================================================
# PATHS
# ================================================================

TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_split"
    / "train_manifest.csv"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_features"
)

QUESTION_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_question_embeddings"
)


# ================================================================
# EXPECTED CONFIGURATION
# ================================================================

EXPECTED_MODEL_PARAMETERS = 1744925


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


VOCABULARIES = {
    "binary": (
        "no",
        "yes",
    ),

    "land_cover": (
        "buildings",
        "trees",
        "low_vegetation",
        "NVG_surface",
        "water",
        "playgrounds",
    ),

    "change_ratio": (
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
    ),

    "change_ratio_types": (
        "0",
        "0_to_10",
        "10_to_20",
        "20_to_30",
        "30_to_40",
        "40_to_50",
        "50_to_60",
        "60_to_70",
        "70_to_80",
    ),
}


# ================================================================
# QUESTION EMBEDDING CACHE
# ================================================================

def load_question_embeddings():

    print()
    print("=" * 72)
    print("LOADING TEMPORAL QUESTION EMBEDDING CACHE")
    print("=" * 72)

    start = time.perf_counter()

    shard_paths = sorted(
        QUESTION_CACHE_DIR.glob(
            "shard_*.pt"
        )
    )

    if not shard_paths:
        raise RuntimeError(
            "No temporal question embedding shards found."
        )

    embedding_by_id = {}

    for shard_path in shard_paths:

        data = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        ids = data["ids"]
        embeddings = data[
            "question_embeddings"
        ]

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"{shard_path.name}: "
                f"unexpected embedding shape "
                f"{tuple(embeddings.shape)}"
            )

        if embeddings.shape[1] != 768:
            raise RuntimeError(
                f"{shard_path.name}: "
                "expected embedding dimension 768."
            )

        if len(ids) != embeddings.shape[0]:
            raise RuntimeError(
                f"{shard_path.name}: "
                "ID/embedding count mismatch."
            )

        for index, question_id in enumerate(ids):

            qid = int(
                question_id
            )

            if qid in embedding_by_id:
                raise RuntimeError(
                    f"Duplicate question ID: {qid}"
                )

            embedding_by_id[qid] = (
                embeddings[index].float()
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        "Cache shards:",
        len(shard_paths),
    )

    print(
        "Cached question IDs:",
        f"{len(embedding_by_id):,}",
    )

    print(
        f"Cache load time: {elapsed:.3f} s"
    )

    return embedding_by_id


# ================================================================
# TARGET ENCODING
# ================================================================

def encode_target(
    question_type,
    answer,
):

    head = QUESTION_TYPE_TO_HEAD.get(
        question_type
    )

    if head is None:
        raise RuntimeError(
            f"Unknown question type: "
            f"{question_type}"
        )

    answer = str(
        answer
    )

    vocabulary = VOCABULARIES[
        head
    ]

    if answer not in vocabulary:
        raise RuntimeError(
            f"Answer '{answer}' is not valid "
            f"for head '{head}'."
        )

    target = vocabulary.index(
        answer
    )

    return head, target


# ================================================================
# ROUTED LOSS
# ================================================================

def compute_routed_loss(
    outputs,
    question_types,
    answers,
):

    losses = []

    routed = {
        "binary": 0,
        "land_cover": 0,
        "change_ratio": 0,
        "change_ratio_types": 0,
    }

    for index, (
        question_type,
        answer,
    ) in enumerate(
        zip(
            question_types,
            answers,
        )
    ):

        head, target = encode_target(
            question_type,
            answer,
        )

        logits = outputs[
            "answer_logits"
        ][head][
            index:index + 1
        ]

        target_tensor = torch.tensor(
            [target],
            dtype=torch.long,
            device=logits.device,
        )

        losses.append(
            F.cross_entropy(
                logits,
                target_tensor,
            )
        )

        routed[head] += 1

    if not losses:
        raise RuntimeError(
            "No losses generated."
        )

    return (
        torch.stack(losses).mean(),
        routed,
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 72)
    print(
        "SATQUERYAI — REAL TEMPORAL "
        "TRAINING-STEP SMOKE TEST"
    )
    print("=" * 72)

    torch.set_num_threads(
        max(
            1,
            min(
                4,
                torch.get_num_threads(),
            ),
        )
    )

    device = torch.device(
        "cpu"
    )

    print()
    print(
        "Device:",
        device,
    )

    print(
        "Torch threads:",
        torch.get_num_threads(),
    )

    # ============================================================
    # QUESTION CACHE
    # ============================================================

    question_embeddings = (
        load_question_embeddings()
    )

    # ============================================================
    # DATASET
    # ============================================================

    print()
    print("=" * 72)
    print("LOADING TRAINING DATA")
    print("=" * 72)

    if not TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            f"Training manifest not found:\n"
            f"{TRAIN_MANIFEST}"
        )

    dataset_start = time.perf_counter()

    dataset = TemporalDOFADataset(
        manifest_path=str(
            TRAIN_MANIFEST
        ),
        feature_dir=str(
            FEATURE_DIR
        ),
    )

    dataset_elapsed = (
        time.perf_counter()
        - dataset_start
    )

    print()
    print(
        "Training QA rows:",
        f"{len(dataset):,}",
    )

    print(
        f"Dataset initialization: "
        f"{dataset_elapsed:.3f} s"
    )

    # ============================================================
    # SELECT ONE SAMPLE PER QUESTION TYPE
    # ============================================================

    desired_types = [
        "change_or_not",
        "increase_or_not",
        "decrease_or_not",
        "smallest_change",
        "largest_change",
        "change_to_what",
        "change_ratio",
        "change_ratio_types",
    ]

    selection_start = time.perf_counter()

    selected_indices = []

    found_types = set()

    for index in range(
        len(dataset)
    ):

        sample = dataset[
            index
        ]

        question_type = str(
            sample["question_type"]
        )

        if (
            question_type in desired_types
            and question_type not in found_types
        ):

            selected_indices.append(
                index
            )

            found_types.add(
                question_type
            )

        if len(found_types) == len(
            desired_types
        ):
            break

    selection_elapsed = (
        time.perf_counter()
        - selection_start
    )

    if len(found_types) != len(
        desired_types
    ):
        raise RuntimeError(
            "Not all question types found."
        )

    print()
    print(
        "Selected real examples:",
        len(selected_indices),
    )

    print(
        f"Sample selection time: "
        f"{selection_elapsed:.3f} s"
    )

    # ============================================================
    # LOAD REAL SAMPLES
    # ============================================================

    samples_start = time.perf_counter()

    samples = [
        dataset[index]
        for index in selected_indices
    ]

    question_vectors = []

    for sample in samples:

        qid = int(
            sample["question_id"]
        )

        if qid not in question_embeddings:
            raise RuntimeError(
                f"Missing embedding for "
                f"question ID {qid}"
            )

        question_vectors.append(
            question_embeddings[qid]
        )

    question_tensor = torch.stack(
        question_vectors
    ).to(
        device=device,
        dtype=torch.float32,
    )

    t1 = torch.stack(
        [
            sample["t1"].float()
            for sample in samples
        ]
    ).to(device)

    t2 = torch.stack(
        [
            sample["t2"].float()
            for sample in samples
        ]
    ).to(device)

    diff = torch.stack(
        [
            sample["diff"].float()
            for sample in samples
        ]
    ).to(device)

    question_types = [
        str(
            sample["question_type"]
        )
        for sample in samples
    ]

    answers = [
        str(
            sample["answer"]
        )
        for sample in samples
    ]

    samples_elapsed = (
        time.perf_counter()
        - samples_start
    )

    print()
    print("REAL TRAINING BATCH")
    print("-" * 72)

    print(
        "T1:",
        tuple(t1.shape),
    )

    print(
        "T2:",
        tuple(t2.shape),
    )

    print(
        "Difference:",
        tuple(diff.shape),
    )

    print(
        "Question:",
        tuple(question_tensor.shape),
    )

    print(
        f"Batch preparation time: "
        f"{samples_elapsed:.3f} s"
    )

    for sample in samples:

        print(
            f"{sample['question_type']:22s}"
            f" qid={int(sample['question_id']):5d}"
            f" answer={sample['answer']}"
        )

    # ============================================================
    # MODEL
    # ============================================================

    print()
    print("=" * 72)
    print("MODEL")
    print("=" * 72)

    model_start = time.perf_counter()

    model = TemporalChangeVQA().to(
        device
    )

    model.train()

    model_init_elapsed = (
        time.perf_counter()
        - model_start
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        parameter_count,
    )

    print(
        f"Model initialization: "
        f"{model_init_elapsed:.3f} s"
    )

    if parameter_count != (
        EXPECTED_MODEL_PARAMETERS
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_MODEL_PARAMETERS:,} "
            f"parameters, found "
            f"{parameter_count:,}."
        )

    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer_start = time.perf_counter()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    optimizer_init_elapsed = (
        time.perf_counter()
        - optimizer_start
    )

    print(
        f"Optimizer initialization: "
        f"{optimizer_init_elapsed:.3f} s"
    )

    # ============================================================
    # TRACK PARAMETER UPDATE
    # ============================================================

    tracked_name = (
        "temporal_fusion.t1_proj.weight"
    )

    parameter_before = (
        model.state_dict()[
            tracked_name
        ].detach().clone()
    )

    # ============================================================
    # FORWARD + LOSS
    # ============================================================

    print()
    print("=" * 72)
    print("FORWARD + LOSS")
    print("=" * 72)

    forward_start = time.perf_counter()

    outputs = model(
        t1=t1,
        t2=t2,
        diff=diff,
        question=question_tensor,
    )

    forward_elapsed = (
        time.perf_counter()
        - forward_start
    )

    loss_start = time.perf_counter()

    loss, routed = (
        compute_routed_loss(
            outputs,
            question_types,
            answers,
        )
    )

    loss_elapsed = (
        time.perf_counter()
        - loss_start
    )

    print(
        f"Forward time: "
        f"{forward_elapsed:.4f} s"
    )

    print(
        f"Loss calculation: "
        f"{loss_elapsed:.4f} s"
    )

    print(
        "Loss:",
        float(loss.item()),
    )

    print(
        "Routed samples:",
        routed,
    )

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Training loss is not finite."
        )

    # ============================================================
    # BACKWARD
    # ============================================================

    print()
    print("=" * 72)
    print("BACKWARD")
    print("=" * 72)

    backward_start = time.perf_counter()

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    backward_elapsed = (
        time.perf_counter()
        - backward_start
    )

    print(
        f"Backward time: "
        f"{backward_elapsed:.4f} s"
    )

    # ============================================================
    # GRADIENT CHECK
    # ============================================================

    gradients_present = 0
    finite_gradients = True

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            continue

        gradients_present += 1

        if not torch.isfinite(
            parameter.grad
        ).all():

            finite_gradients = False

            print(
                "Non-finite gradient:",
                name
            )

    print()
    print(
        "Parameters with gradients:",
        gradients_present,
    )

    if gradients_present == 0:
        raise RuntimeError(
            "No gradients were produced."
        )

    if not finite_gradients:
        raise RuntimeError(
            "Non-finite gradient detected."
        )

    # ============================================================
    # OPTIMIZER STEP
    # ============================================================

    print()
    print("=" * 72)
    print("OPTIMIZER STEP")
    print("=" * 72)

    step_start = time.perf_counter()

    optimizer.step()

    step_elapsed = (
        time.perf_counter()
        - step_start
    )

    parameter_after = (
        model.state_dict()[
            tracked_name
        ].detach().clone()
    )

    parameter_delta = (
        parameter_after
        - parameter_before
    ).abs().max().item()

    print(
        f"Optimizer step time: "
        f"{step_elapsed:.4f} s"
    )

    print(
        "Maximum parameter update:",
        f"{parameter_delta:.10f}",
    )

    if parameter_delta <= 0.0:
        raise RuntimeError(
            "No parameter update occurred."
        )

    # ============================================================
    # TOTAL TRAINING STEP TIME
    # ============================================================

    total_step_time = (
        forward_elapsed
        + loss_elapsed
        + backward_elapsed
        + step_elapsed
    )

    effective_examples = len(
        samples
    )

    examples_per_second = (
        effective_examples
        / total_step_time
    )

    print()
    print("=" * 72)
    print("TIMING SUMMARY")
    print("=" * 72)

    print(
        f"Forward               : "
        f"{forward_elapsed:.4f} s"
    )

    print(
        f"Loss                  : "
        f"{loss_elapsed:.4f} s"
    )

    print(
        f"Backward              : "
        f"{backward_elapsed:.4f} s"
    )

    print(
        f"Optimizer step        : "
        f"{step_elapsed:.4f} s"
    )

    print(
        f"Total training step   : "
        f"{total_step_time:.4f} s"
    )

    print(
        f"Examples / second     : "
        f"{examples_per_second:.4f}"
    )

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("=" * 72)
    print("TRAINING-STEP RESULTS")
    print("=" * 72)

    print(
        "Real temporal inputs       : PASS"
    )

    print(
        "Real question embeddings   : PASS"
    )

    print(
        "All question-type routes   : PASS"
    )

    print(
        "Forward pass               : PASS"
    )

    print(
        "Finite loss                : PASS"
    )

    print(
        "Backward pass              : PASS"
    )

    print(
        "Finite gradients           : PASS"
    )

    print(
        "Optimizer update           : PASS"
    )

    print(
        "Timing measurement         : PASS"
    )

    print()
    print("=" * 72)
    print(
        "TEMPORAL REAL TRAINING-STEP "
        "SMOKE TEST PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()