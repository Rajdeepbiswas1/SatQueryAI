from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, r"D:\SatQueryAI\src")

from data.bigearthnet_dataset import BigEarthNetDOFADataset
from data.dataloaders import make_bigearthnet_loader
from models.binary_vqa import BinaryVQAModel


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
    )

    loader = make_bigearthnet_loader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    # --------------------------------------------------------
    # Tokenize actual questions
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    encoded = tokenizer(
        batch["question"],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = BinaryVQAModel(
        text_model_name="distilbert-base-uncased",
        freeze_text_encoder=True,
    ).to(device)

    model.eval()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    with torch.no_grad():

        logits = model(
            batch["s1"].to(device),
            batch["s2"].to(device),
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )

    probabilities = F.softmax(
        logits,
        dim=-1,
    )

    predictions = probabilities.argmax(
        dim=-1
    )

    print("Questions:", len(batch["question"]))
    print("Logits:", tuple(logits.shape))
    print("Probabilities:", tuple(probabilities.shape))
    print("Predictions:", predictions.tolist())

    print("\nExamples:")

    for i in range(len(batch["question"])):
        prediction = (
            "yes"
            if predictions[i].item() == 1
            else "no"
        )

        print(
            f"{i+1}. {batch['question'][i]}"
        )
        print(
            f"   Target: {batch['answer'][i]}"
        )
        print(
            f"   Initial prediction: {prediction}"
        )
        print(
            f"   Probabilities: "
            f"no={probabilities[i,0].item():.4f}, "
            f"yes={probabilities[i,1].item():.4f}"
        )

    # --------------------------------------------------------
    # Numerical checks
    # --------------------------------------------------------
    assert logits.shape == (8, 2)
    assert probabilities.shape == (8, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(probabilities).all()

    print()
    print("REAL BINARY VQA FORWARD TEST PASSED")


if __name__ == "__main__":
    main()