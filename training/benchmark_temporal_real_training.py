from pathlib import Path
import sys
import time
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(r"D:\SatQueryAI")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.temporal_dataset import TemporalDOFADataset
from src.models.temporal_change_vqa import TemporalChangeVQA


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

BATCH_SIZE = 8
WARMUP_STEPS = 5
TIMED_STEPS = 30

EXPECTED_PARAMETERS = 1744925


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


def load_question_embedding_matrix():

    print()
    print("=" * 72)
    print("LOADING QUESTION EMBEDDINGS")
    print("=" * 72)

    start = time.perf_counter()

    shards = sorted(
        QUESTION_CACHE_DIR.glob("shard_*.pt")
    )

    if not shards:
        raise RuntimeError(
            "No temporal question embedding shards found."
        )

    records = []

    for shard_path in shards:

        data = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        ids = [
            int(x)
            for x in data["ids"]
        ]

        embeddings = data[
            "question_embeddings"
        ].float()

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Invalid embedding tensor in "
                f"{shard_path.name}"
            )

        if embeddings.shape[1] != 768:
            raise RuntimeError(
                f"Expected 768 dimensions in "
                f"{shard_path.name}"
            )

        if len(ids) != embeddings.shape[0]:
            raise RuntimeError(
                f"ID/embedding mismatch in "
                f"{shard_path.name}"
            )

        for qid, embedding in zip(
            ids,
            embeddings,
        ):
            records.append(
                (
                    qid,
                    embedding,
                )
            )

    records.sort(
        key=lambda x: x[0]
    )

    question_ids = [
        qid
        for qid, _ in records
    ]

    embeddings = torch.stack(
        [
            embedding
            for _, embedding in records
        ]
    )

    expected_ids = list(
        range(
            len(question_ids)
        )
    )

    if question_ids != expected_ids:
        raise RuntimeError(
            "Question IDs are not a complete "
            "contiguous 0..N-1 sequence."
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        "Question IDs:",
        len(question_ids)
    )

    print(
        "Embedding matrix:",
        tuple(embeddings.shape)
    )

    print(
        f"Load time: {elapsed:.3f} s"
    )

    return embeddings


def encode_batch_targets(
    question_types,
    answers,
):

    grouped_indices = {
        "binary": [],
        "land_cover": [],
        "change_ratio": [],
        "change_ratio_types": [],
    }

    grouped_targets = {
        "binary": [],
        "land_cover": [],
        "change_ratio": [],
        "change_ratio_types": [],
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

        head = QUESTION_TYPE_TO_HEAD.get(
            question_type
        )

        if head is None:
            raise RuntimeError(
                f"Unknown question type: "
                f"{question_type}"
            )

        vocabulary = VOCABULARIES[
            head
        ]

        answer = str(answer)

        if answer not in vocabulary:
            raise RuntimeError(
                f"Invalid answer '{answer}' "
                f"for head '{head}'."
            )

        target = vocabulary.index(
            answer
        )

        grouped_indices[
            head
        ].append(index)

        grouped_targets[
            head
        ].append(target)

    return (
        grouped_indices,
        grouped_targets,
    )


def compute_loss(
    outputs,
    question_types,
    answers,
):

    batch_size = len(
        question_types
    )

    grouped_indices, grouped_targets = (
        encode_batch_targets(
            question_types,
            answers,
        )
    )

    total_loss = None

    counts = {}

    for head in grouped_indices:

        indices = grouped_indices[
            head
        ]

        counts[head] = len(
            indices
        )

        if not indices:
            continue

        index_tensor = torch.tensor(
            indices,
            dtype=torch.long,
            device=outputs[
                "answer_logits"
            ][head].device,
        )

        target_tensor = torch.tensor(
            grouped_targets[head],
            dtype=torch.long,
            device=index_tensor.device,
        )

        logits = outputs[
            "answer_logits"
        ][head].index_select(
            0,
            index_tensor,
        )

        head_loss = F.cross_entropy(
            logits,
            target_tensor,
            reduction="sum",
        )

        if total_loss is None:
            total_loss = head_loss
        else:
            total_loss = (
                total_loss
                + head_loss
            )

    if total_loss is None:
        raise RuntimeError(
            "No head received training samples."
        )

    total_loss = (
        total_loss
        / batch_size
    )

    return total_loss, counts


def prepare_batch(
    batch,
    question_embeddings,
    device,
):

    qids = batch[
        "question_id"
    ].long()

    question = (
        question_embeddings[
            qids
        ]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    t1 = batch[
        "t1"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    t2 = batch[
        "t2"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    diff = batch[
        "diff"
    ].to(
        device=device,
        dtype=torch.float32,
    )

    question_types = [
        str(x)
        for x in batch[
            "question_type"
        ]
    ]

    answers = [
        str(x)
        for x in batch[
            "answer"
        ]
    ]

    return (
        t1,
        t2,
        diff,
        question,
        question_types,
        answers,
    )


def training_step(
    model,
    optimizer,
    batch,
    question_embeddings,
    device,
):

    (
        t1,
        t2,
        diff,
        question,
        question_types,
        answers,
    ) = prepare_batch(
        batch,
        question_embeddings,
        device,
    )

    outputs = model(
        t1=t1,
        t2=t2,
        diff=diff,
        question=question,
    )

    loss, counts = compute_loss(
        outputs,
        question_types,
        answers,
    )

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Non-finite loss."
        )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    for parameter in model.parameters():

        if (
            parameter.requires_grad
            and parameter.grad is not None
        ):

            if not torch.isfinite(
                parameter.grad
            ).all():

                raise RuntimeError(
                    "Non-finite gradient."
                )

    optimizer.step()

    return (
        loss,
        counts,
    )


def main():

    print("=" * 72)
    print(
        "SATQUERYAI — REAL TEMPORAL "
        "TRAINING THROUGHPUT BENCHMARK"
    )
    print("=" * 72)

    torch.manual_seed(42)

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

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Warmup steps:",
        WARMUP_STEPS,
    )

    print(
        "Timed steps:",
        TIMED_STEPS,
    )

    # ------------------------------------------------------------
    # Question embeddings
    # ------------------------------------------------------------

    question_embeddings = (
        load_question_embedding_matrix()
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("LOADING TRAINING DATASET")
    print("=" * 72)

    dataset_start = time.perf_counter()

    dataset = TemporalDOFADataset(
        manifest_path=str(
            TRAIN_MANIFEST
        ),
        feature_dir=str(
            FEATURE_DIR
        ),
    )

    dataset_init_time = (
        time.perf_counter()
        - dataset_start
    )

    print(
        "Training QA rows:",
        f"{len(dataset):,}",
    )

    print(
        f"Dataset initialization: "
        f"{dataset_init_time:.3f} s"
    )

    # ------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    batches_per_epoch = len(
        loader
    )

    print(
        "Batches per epoch:",
        batches_per_epoch,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = TemporalChangeVQA().to(
        device
    )

    model.train()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        "Trainable parameters:",
        parameter_count,
    )

    if parameter_count != (
        EXPECTED_PARAMETERS
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PARAMETERS:,} "
            f"parameters, found "
            f"{parameter_count:,}."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("WARMUP")
    print("=" * 72)

    iterator = iter(
        loader
    )

    for step in range(
        WARMUP_STEPS
    ):

        try:
            batch = next(
                iterator
            )
        except StopIteration:
            iterator = iter(
                loader
            )
            batch = next(
                iterator
            )

        training_step(
            model,
            optimizer,
            batch,
            question_embeddings,
            device,
        )

        print(
            f"Warmup step "
            f"{step + 1}/{WARMUP_STEPS}"
        )

    # ------------------------------------------------------------
    # Timed benchmark
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("TIMED REAL TRAINING")
    print("=" * 72)

    iterator = iter(
        loader
    )

    total_data_time = 0.0
    total_compute_time = 0.0
    total_time = 0.0

    completed = 0
    examples_processed = 0

    while completed < TIMED_STEPS:

        data_start = time.perf_counter()

        try:
            batch = next(
                iterator
            )
        except StopIteration:
            iterator = iter(
                loader
            )
            batch = next(
                iterator
            )

        data_time = (
            time.perf_counter()
            - data_start
        )

        batch_size_actual = (
            batch["question_id"].shape[0]
        )

        compute_start = time.perf_counter()

        loss, counts = training_step(
            model,
            optimizer,
            batch,
            question_embeddings,
            device,
        )

        compute_time = (
            time.perf_counter()
            - compute_start
        )

        step_time = (
            data_time
            + compute_time
        )

        total_data_time += data_time
        total_compute_time += compute_time
        total_time += step_time

        completed += 1

        examples_processed += (
            batch_size_actual
        )

        print(
            f"Step {completed:2d}/{TIMED_STEPS} | "
            f"data={data_time:.3f}s | "
            f"compute={compute_time:.3f}s | "
            f"total={step_time:.3f}s | "
            f"loss={float(loss.item()):.4f} | "
            f"batch={batch_size_actual}"
        )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    avg_data_time = (
        total_data_time
        / completed
    )

    avg_compute_time = (
        total_compute_time
        / completed
    )

    avg_step_time = (
        total_time
        / completed
    )

    examples_per_second = (
        examples_processed
        / total_time
    )

    epoch_seconds = (
        len(dataset)
        / examples_per_second
    )

    epoch_minutes = (
        epoch_seconds
        / 60.0
    )

    epoch_hours = (
        epoch_minutes
        / 60.0
    )

    print()
    print("=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)

    print()
    print(
        f"Timed steps             : "
        f"{completed}"
    )

    print(
        f"Examples processed      : "
        f"{examples_processed}"
    )

    print(
        f"Average data time       : "
        f"{avg_data_time:.4f} s"
    )

    print(
        f"Average compute time    : "
        f"{avg_compute_time:.4f} s"
    )

    print(
        f"Average total step      : "
        f"{avg_step_time:.4f} s"
    )

    print(
        f"Examples / second       : "
        f"{examples_per_second:.4f}"
    )

    print(
        f"Estimated epoch         : "
        f"{epoch_seconds:.1f} s"
    )

    print(
        f"Estimated epoch         : "
        f"{epoch_minutes:.2f} min"
    )

    print(
        f"Estimated epoch         : "
        f"{epoch_hours:.2f} h"
    )

    print()
    print(
        "Training data: REAL"
    )

    print(
        "Question embeddings: REAL"
    )

    print(
        "Answers: REAL"
    )

    print(
        "Loss targets: REAL"
    )

    print(
        "No checkpoint saved."
    )

    print(
        "No dataset modified."
    )

    print()
    print("=" * 72)
    print(
        "REAL TEMPORAL TRAINING "
        "BENCHMARK COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()