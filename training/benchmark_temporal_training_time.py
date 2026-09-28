from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.temporal_dataset import TemporalDOFADataset
from src.models.temporal_change_vqa import TemporalChangeVQA


ROOT = r"D:\SatQueryAI"

MANIFEST = (
    ROOT
    + r"\data\processed\temporal_manifest.csv"
)

FEATURE_DIR = (
    ROOT
    + r"\data\processed\temporal_features"
)

BATCH_SIZE = 8
BENCHMARK_EXAMPLES = 512
WARMUP_STEPS = 5
TIMED_STEPS = 30


def main() -> None:
    print("=" * 72)
    print("SATQUERYAI — TEMPORAL CPU TRAINING BENCHMARK")
    print("=" * 72)

    torch.set_num_threads(
        max(1, min(4, torch.get_num_threads()))
    )

    device = torch.device("cpu")

    print()
    print("Device:", device)
    print("Torch threads:", torch.get_num_threads())
    print("Batch size:", BATCH_SIZE)
    print("Benchmark examples:", BENCHMARK_EXAMPLES)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    dataset = TemporalDOFADataset(
        manifest_path=MANIFEST,
        feature_dir=FEATURE_DIR,
    )

    count = min(
        BENCHMARK_EXAMPLES,
        len(dataset),
    )

    subset = Subset(
        dataset,
        list(range(count)),
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print()
    print("Dataset rows available:", len(dataset))
    print("Rows benchmarked:", count)
    print(
        "Benchmark batches:",
        len(loader),
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = TemporalChangeVQA().to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Trainable parameters:",
        parameter_count,
    )

    # ------------------------------------------------------------
    # One training step
    #
    # The current temporal dataset stores question text rather
    # than precomputed question vectors.
    #
    # For this timing benchmark only, random question vectors
    # are used so we measure the temporal model's training cost
    # without introducing text-encoder overhead.
    # ------------------------------------------------------------

    def train_step(batch):
        t1 = batch["t1"].to(
            device=device,
            dtype=torch.float32,
        )

        t2 = batch["t2"].to(
            device=device,
            dtype=torch.float32,
        )

        diff = batch["diff"].to(
            device=device,
            dtype=torch.float32,
        )

        question = torch.randn(
            t1.size(0),
            768,
            dtype=torch.float32,
            device=device,
        )

        outputs = model(
            t1=t1,
            t2=t2,
            diff=diff,
            question=question,
        )

        binary_logits = outputs[
            "answer_logits"
        ]["binary"]

        ratio_logits = outputs[
            "answer_logits"
        ]["change_ratio"]

        ratio_type_logits = outputs[
            "answer_logits"
        ]["change_ratio_types"]

        land_cover_logits = outputs[
            "answer_logits"
        ]["land_cover"]

        target_binary = torch.randint(
            0,
            2,
            (t1.size(0),),
            device=device,
        )

        target_ratio = torch.randint(
            0,
            11,
            (t1.size(0),),
            device=device,
        )

        target_ratio_type = torch.randint(
            0,
            7,
            (t1.size(0),),
        ).to(device)

        target_land = torch.randint(
            0,
            6,
            (t1.size(0),),
            device=device,
        )

        loss = (
            F.cross_entropy(
                binary_logits,
                target_binary,
            )
            + F.cross_entropy(
                ratio_logits,
                target_ratio,
            )
            + F.cross_entropy(
                ratio_type_logits,
                target_ratio_type,
            )
            + F.cross_entropy(
                land_cover_logits,
                target_land,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    print()
    print("Warmup...")

    iterator = iter(loader)

    for _ in range(WARMUP_STEPS):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        train_step(batch)

    # ------------------------------------------------------------
    # Timed training
    # ------------------------------------------------------------

    print("Timing training steps...")

    iterator = iter(loader)

    start = time.perf_counter()

    completed = 0

    for _ in range(TIMED_STEPS):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        train_step(batch)
        completed += 1

    elapsed = (
        time.perf_counter()
        - start
    )

    seconds_per_step = (
        elapsed / completed
    )

    examples_processed = (
        completed * BATCH_SIZE
    )

    examples_per_second = (
        examples_processed / elapsed
    )

    total_rows = len(dataset)

    estimated_epoch_seconds = (
        total_rows
        / examples_per_second
    )

    estimated_epoch_minutes = (
        estimated_epoch_seconds / 60
    )

    estimated_epoch_hours = (
        estimated_epoch_minutes / 60
    )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)

    print()
    print(
        f"Timed steps          : {completed}"
    )

    print(
        f"Examples processed   : "
        f"{examples_processed}"
    )

    print(
        f"Total timed time     : "
        f"{elapsed:.2f} s"
    )

    print(
        f"Seconds per step     : "
        f"{seconds_per_step:.3f} s"
    )

    print(
        f"Examples / second    : "
        f"{examples_per_second:.3f}"
    )

    print(
        f"Estimated full epoch : "
        f"{estimated_epoch_seconds:.1f} s"
    )

    print(
        f"Estimated full epoch : "
        f"{estimated_epoch_minutes:.2f} min"
    )

    print(
        f"Estimated full epoch : "
        f"{estimated_epoch_hours:.2f} h"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This is a model-training throughput benchmark."
    )
    print(
        "Question-embedding generation is not included."
    )
    print(
        "No model checkpoint was saved."
    )
    print(
        "No temporal dataset was modified."
    )

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()