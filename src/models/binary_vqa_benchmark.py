from __future__ import annotations

import sys
import time

import torch
from transformers import AutoTokenizer
from torch.optim import AdamW

sys.path.insert(0, r"D:\SatQueryAI\src")

from data.bigearthnet_dataset import BigEarthNetDOFADataset
from data.dataloaders import make_bigearthnet_loader
from models.binary_vqa import BinaryVQAModel


ANSWER_TO_LABEL = {
    "no": 0,
    "yes": 1,
}


def main() -> None:
    device = torch.device("cpu")

    # --------------------------------------------------------
    # Dataset: ONLY binary training split
    # --------------------------------------------------------
    dataset = BigEarthNetDOFADataset(
        annotations_path=(
            r"D:\SatQueryAI\data\processed"
            r"\binary_vqa_split.parquet"
        ),
        feature_dir=(
            r"D:\SatQueryAI\data\processed"
            r"\dofa_features_download\dofa_features"
        ),
        split="train",
        task_type="binary",
        model_split="train",
    )

    loader = make_bigearthnet_loader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = BinaryVQAModel(
        text_model_name="distilbert-base-uncased",
        freeze_text_encoder=True,
    ).to(device)

    model.train()

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=1e-3,
        weight_decay=1e-4,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    print(
        "Trainable parameters:",
        sum(p.numel() for p in trainable_parameters),
    )

    print(
        "Frozen parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if not p.requires_grad
        ),
    )

    print()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------
    total_examples = 0
    total_correct = 0
    total_loss = 0.0

    start_time = time.perf_counter()

    for step, batch in enumerate(loader):

        if step >= 100:
            break

        encoded = tokenizer(
            batch["question"],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        labels = torch.tensor(
            [
                ANSWER_TO_LABEL[
                    answer.strip().lower()
                ]
                for answer in batch["answer"]
            ],
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------
        logits = model(
            batch["s1"].to(device),
            batch["s2"].to(device),
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )

        loss = loss_fn(
            logits,
            labels,
        )

        predictions = logits.argmax(dim=-1)

        correct = (
            predictions == labels
        ).sum().item()

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------
        optimizer.zero_grad()
        loss.backward()

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------
        optimizer.step()

        batch_size = labels.shape[0]

        total_examples += batch_size
        total_correct += correct
        total_loss += loss.item() * batch_size

        if (step + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time

            print(
                f"Step {step + 1:03d}/100 | "
                f"Loss: {loss.item():.4f} | "
                f"Batch Acc: {correct / batch_size:.3f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - start_time

    print()
    print("==============================")
    print("100-BATCH BENCHMARK")
    print("==============================")
    print("Examples:", total_examples)
    print("Average loss:", total_loss / total_examples)
    print("Accuracy:", total_correct / total_examples)
    print("Time:", round(elapsed, 2), "seconds")
    print(
        "Time/batch:",
        round(elapsed / 100, 3),
        "seconds"
    )
    print(
        "Examples/sec:",
        round(total_examples / elapsed, 2)
    )

    print()
    print("BINARY VQA BENCHMARK PASSED")


if __name__ == "__main__":
    main()