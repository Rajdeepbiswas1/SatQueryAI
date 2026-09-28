from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "distilbert-base-uncased"


def main() -> None:
    device = torch.device("cpu")

    # --------------------------------------------------------
    # Example questions from our actual BigEarthNet data
    # --------------------------------------------------------
    questions = [
        "Does any mixed forest touch pastures in the image?",
        "Are there water bodies in the image?",
        "Do the forest areas cover a large portion of the scene?",
    ]

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    print("Input IDs:", tuple(encoded["input_ids"].shape))
    print("Attention mask:", tuple(encoded["attention_mask"].shape))

    # --------------------------------------------------------
    # Text encoder
    # --------------------------------------------------------
    text_encoder = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)

    text_encoder.eval()

    # For this first test, freeze DistilBERT.
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)

    # --------------------------------------------------------
    # Encode questions
    # --------------------------------------------------------
    with torch.no_grad():
        output = text_encoder(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )

    # DistilBERT hidden states:
    # [batch, sequence_length, hidden_size]
    hidden_states = output.last_hidden_state

    print("Hidden states:", tuple(hidden_states.shape))
    print("Hidden dtype:", hidden_states.dtype)
    print("Finite:", bool(torch.isfinite(hidden_states).all()))

    # --------------------------------------------------------
    # Create one vector representing each question.
    #
    # We use masked mean pooling so padding tokens don't
    # contribute to the average.
    # --------------------------------------------------------
    mask = encoded["attention_mask"].to(device).unsqueeze(-1)

    masked_hidden = hidden_states * mask

    summed = masked_hidden.sum(dim=1)

    lengths = mask.sum(dim=1).clamp(min=1)

    question_embeddings = summed / lengths

    print(
        "Question embeddings:",
        tuple(question_embeddings.shape),
    )
    print(
        "Question embedding dtype:",
        question_embeddings.dtype,
    )
    print(
        "Question embeddings finite:",
        bool(torch.isfinite(question_embeddings).all()),
    )

    print()
    print("QUESTION ENCODER TEST PASSED")


if __name__ == "__main__":
    main()