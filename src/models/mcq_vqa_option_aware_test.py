import sys

import torch

sys.path.insert(0, "src")

from models.mcq_vqa_option_aware import (
    MCQOptionAwareVQA,
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
NUM_TOKENS = 196
FEATURE_DIM = 768
NUM_OPTIONS = 4


# ============================================================
# MODEL
# ============================================================

model = MCQOptionAwareVQA()

model.eval()


# ============================================================
# SYNTHETIC INPUTS
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

options = torch.randn(
    BATCH_SIZE,
    NUM_OPTIONS,
    FEATURE_DIM,
)


# ============================================================
# FORWARD
# ============================================================

with torch.no_grad():

    logits = model(
        s1,
        s2,
        question,
        options,
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predictions = logits.argmax(
        dim=1
    )


# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("OPTION-AWARE MCQ FORWARD TEST")
print("=" * 60)

print(
    "S1:",
    tuple(s1.shape)
)

print(
    "S2:",
    tuple(s2.shape)
)

print(
    "Question:",
    tuple(question.shape)
)

print(
    "Options:",
    tuple(options.shape)
)

print(
    "Logits:",
    tuple(logits.shape)
)

print(
    "Probabilities:",
    tuple(probabilities.shape)
)

print(
    "Predictions:",
    predictions.tolist()
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
    "Probability sums:",
    probabilities.sum(dim=1)
)


# ============================================================
# VALIDATION
# ============================================================

if logits.shape != (
    BATCH_SIZE,
    NUM_OPTIONS,
):
    raise RuntimeError(
        "Incorrect logits shape."
    )

if not torch.isfinite(
    logits
).all():

    raise RuntimeError(
        "Logits contain NaN/Inf."
    )

if not torch.allclose(
    probabilities.sum(dim=1),
    torch.ones(BATCH_SIZE),
    atol=1e-5,
):

    raise RuntimeError(
        "Probabilities do not sum to 1."
    )


print(
    "\nOPTION-AWARE MCQ FORWARD TEST PASSED"
)