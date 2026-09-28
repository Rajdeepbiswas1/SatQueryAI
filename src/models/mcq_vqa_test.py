import sys

import torch

sys.path.insert(0, "src")

from models.mcq_vqa import MCQVQAModel


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
NUM_TOKENS = 196
FEATURE_DIM = 768


# ============================================================
# CREATE MODEL
# ============================================================

model = MCQVQAModel()

model.eval()


# ============================================================
# FAKE INPUTS
# ============================================================

s1 = torch.randn(
    BATCH_SIZE,
    NUM_TOKENS,
    FEATURE_DIM,
)

s2 = torch.randn(
    BATCH_SIZE,
    NUM_TOKENS,
    FEATURE_DIM,
)

question = torch.randn(
    BATCH_SIZE,
    FEATURE_DIM,
)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    logits = model(
        s1,
        s2,
        question,
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predictions = logits.argmax(
        dim=1
    )


# ============================================================
# CHECKS
# ============================================================

print("=" * 60)
print("MCQ VQA FORWARD TEST")
print("=" * 60)

print(
    "S1 shape:",
    tuple(s1.shape)
)

print(
    "S2 shape:",
    tuple(s2.shape)
)

print(
    "Question shape:",
    tuple(question.shape)
)

print(
    "Logits shape:",
    tuple(logits.shape)
)

print(
    "Probabilities shape:",
    tuple(probabilities.shape)
)

print(
    "Predictions shape:",
    tuple(predictions.shape)
)

print(
    "Predictions:",
    predictions.tolist()
)

print(
    "Probabilities:"
)

print(
    probabilities
)

print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)

print(
    "\nLogits finite:",
    torch.isfinite(logits).all().item()
)

print(
    "Probabilities finite:",
    torch.isfinite(
        probabilities
    ).all().item()
)

print(
    "Probability sums:"
)

print(
    probabilities.sum(dim=1)
)

# Every row should sum to approximately 1.
if not torch.allclose(
    probabilities.sum(dim=1),
    torch.ones(BATCH_SIZE),
    atol=1e-5,
):
    raise RuntimeError(
        "Probability rows do not sum to 1."
    )

if logits.shape != (
    BATCH_SIZE,
    4,
):
    raise RuntimeError(
        f"Expected logits shape "
        f"({BATCH_SIZE}, 4), "
        f"got {tuple(logits.shape)}"
    )

print("\nMCQ VQA FORWARD TEST PASSED")