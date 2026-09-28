from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer

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
    # Dataset
    # --------------------------------------------------------
    dataset = BigEarthNetDOFADataset(
        annotations_path=(
            r"D:\SatQueryAI\data\processed"
            r"\training_annotations.parquet"
        ),
        feature_dir=(
            r"D:\SatQueryAI\data\processed"
            r"\dofa_features_download\dofa_features"
        ),
        split="train",
        task_type="binary",
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

    # --------------------------------------------------------
    # Optimizer
    #
    # Only trainable parameters are passed.
    # DistilBERT parameters are frozen.
    # --------------------------------------------------------
    optimizer = AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=1e-3,
        weight_decay=1e-4,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    print("Trainable parameters:", sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ))

    print("Frozen parameters:", sum(
        p.numel()
        for p in model.parameters()
        if not p.requires_grad
    ))

    print()

    # --------------------------------------------------------
    # Run 10 real training steps
    # --------------------------------------------------------
    for step, batch in enumerate(loader):

        if step >= 10:
            break

        questions = batch["question"]

        encoded = tokenizer(
            questions,
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

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------
        loss = loss_fn(
            logits,
            labels,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------
        predictions = logits.argmax(dim=-1)

        accuracy = (
            predictions == labels
        ).float().mean().item()

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------
        optimizer.zero_grad()

        loss.backward()

        # ----------------------------------------------------
        # Gradient check
        # ----------------------------------------------------
        gradients_ok = True

        for parameter in model.parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    gradients_ok = False
                    break

                if not torch.isfinite(parameter.grad).all():
                    gradients_ok = False
                    break

        assert gradients_ok

        # ----------------------------------------------------
        # Parameter update
        # ----------------------------------------------------
        optimizer.step()

        print(
            f"Step {step + 1:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.3f}"
        )

        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)

    print()
    print("REAL BINARY TRAINING TEST PASSED")


if __name__ == "__main__":
    main()