import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.data.grounding_dataset import GroundingDataset


# ============================================================
# Configuration
# ============================================================

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-2

torch.manual_seed(SEED)


# ============================================================
# Load grounding dataset
# ============================================================

print("=" * 70)
print("GROUNDING TEXT LINEAR PROBE")
print("=" * 70)

dataset = GroundingDataset()

df = dataset.df

print(f"Dataset rows: {len(df):,}")

# ------------------------------------------------------------
# Find reference examples
# ------------------------------------------------------------

reference_indices = [
    i
    for i, category in enumerate(
        df["category"].tolist()
    )
    if category == "reference"
]

print(
    f"Reference rows: {len(reference_indices):,}"
)


# ============================================================
# Build phrase -> embedding mapping
# ============================================================

# We previously established that there are 243 unique
# reference phrases.
#
# Here we recover one embedding for every unique phrase and
# assign a coarse semantic category to it.
#
# The category labels are deliberately based on the lexical
# content of the reference phrase. This is a diagnostic, not
# a production classifier.
# ============================================================

def clean_query(text):
    text = str(text).strip()

    if "<ref>" in text and "</ref>" in text:

        start = text.find("<ref>") + len("<ref>")
        end = text.find("</ref>", start)

        if end > start:
            return text[start:end].strip()

    return text


def infer_category(query):
    """
    Assign a broad land-cover category from the actual phrase.

    Priority matters because some phrases may contain multiple
    related words.
    """

    q = query.lower()

    category_rules = [
        ("urban", ["urban fabric"]),
        ("water", [
            "marine waters",
            "inland waters",
            "water",
            "waters",
        ]),
        ("forest", [
            "mixed forest",
            "broad-leaved forest",
            "coniferous forest",
            "forest",
        ]),
        ("pastures", [
            "pastures",
            "pasture",
        ]),
        ("crops", [
            "permanent crops",
            "arable",
            "crop",
            "crops",
        ]),
        ("grassland", [
            "grassland",
        ]),
        ("shrub", [
            "shrub",
            "sclerophyllous",
        ]),
        ("industrial", [
            "industrial",
        ]),
        ("vineyard", [
            "vineyard",
        ]),
        ("orchard", [
            "orchard",
        ]),
    ]

    for category, keywords in category_rules:

        if any(
            keyword in q
            for keyword in keywords
        ):
            return category

    return "other"


# ------------------------------------------------------------
# Recover unique reference phrases
# ------------------------------------------------------------

phrase_to_embedding = {}

for idx in reference_indices:

    query = clean_query(
        df.iloc[idx]["input"]
    )

    if query in phrase_to_embedding:
        continue

    item = dataset[idx]

    embedding = item[
        "text_embedding"
    ].float()

    phrase_to_embedding[query] = embedding


print(
    f"Unique reference phrases recovered: "
    f"{len(phrase_to_embedding):,}"
)


# ============================================================
# Inspect category distribution
# ============================================================

phrases = []
embeddings = []
labels = []

categories = {}

for query, embedding in phrase_to_embedding.items():

    category = infer_category(query)

    phrases.append(query)
    embeddings.append(embedding)
    labels.append(category)

    categories[category] = (
        categories.get(category, 0) + 1
    )


print()
print("=" * 70)
print("CATEGORY DISTRIBUTION")
print("=" * 70)

for category, count in sorted(
    categories.items(),
    key=lambda x: (-x[1], x[0]),
):

    print(
        f"{category:20s} : {count}"
    )


# ============================================================
# Remove categories occurring only once
# ============================================================

# A linear classifier cannot meaningfully learn a class from
# one example, so for this diagnostic we retain categories
# having at least two phrases.

valid_categories = {
    category
    for category, count in categories.items()
    if count >= 2
}

filtered = [
    i
    for i, category in enumerate(labels)
    if category in valid_categories
]

embeddings = [
    embeddings[i]
    for i in filtered
]

labels = [
    labels[i]
    for i in filtered
]

phrases = [
    phrases[i]
    for i in filtered
]

category_names = sorted(
    valid_categories
)

category_to_id = {
    category: i
    for i, category in enumerate(category_names)
}

y = torch.tensor(
    [
        category_to_id[c]
        for c in labels
    ],
    dtype=torch.long,
)

X = torch.stack(
    embeddings
)


print()
print(
    f"Examples after filtering: {len(X):,}"
)

print(
    f"Number of classes:          {len(category_names)}"
)

print(
    f"Embedding dimension:        {X.shape[1]}"
)


# ============================================================
# Train/test split
# ============================================================

generator = torch.Generator().manual_seed(
    SEED
)

full_dataset = TensorDataset(
    X,
    y,
)

train_size = int(
    0.80 * len(full_dataset)
)

test_size = (
    len(full_dataset)
    - train_size
)

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=generator,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# Linear classifier
# ============================================================

model = nn.Linear(
    X.shape[1],
    len(category_names),
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# Training
# ============================================================

print()
print("=" * 70)
print("TRAINING LINEAR PROBE")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in train_loader:

        logits = model(xb)

        loss = criterion(
            logits,
            yb,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item()
            * len(xb)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == yb
        ).sum().item()

        total += len(xb)

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        train_loss = (
            total_loss / total
        )

        train_acc = (
            correct / total
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        model.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():

            for xb, yb in test_loader:

                logits = model(xb)

                predictions = (
                    logits.argmax(dim=1)
                )

                test_correct += (
                    predictions == yb
                ).sum().item()

                test_total += len(xb)

        test_acc = (
            test_correct / test_total
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc:  {test_acc:.4f} | "
            f"Loss: {train_loss:.4f}"
        )


# ============================================================
# Final evaluation
# ============================================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for xb, yb in test_loader:

        logits = model(xb)

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == yb
        ).sum().item()

        total += len(xb)

final_accuracy = (
    correct / total
)


# ============================================================
# Baseline accuracy
# ============================================================

class_counts = torch.bincount(y)

majority_accuracy = (
    class_counts.max().item()
    / len(y)
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("LINEAR PROBE RESULTS")
print("=" * 70)

print(
    f"Final test accuracy: "
    f"{final_accuracy:.4f}"
)

print(
    f"Majority baseline:   "
    f"{majority_accuracy:.4f}"
)

print()

if final_accuracy >= 0.80:

    print(
        "INTERPRETATION: STRONG SEMANTIC SEPARATION"
    )

    print(
        "The pooled DistilBERT embeddings already contain"
    )

    print(
        "substantial fine-grained category information."
    )

    print(
        "The grounding failure is therefore primarily"
    )

    print(
        "a visual grounding/binding problem."
    )

elif final_accuracy >= 0.60:

    print(
        "INTERPRETATION: MODERATE SEMANTIC SEPARATION"
    )

    print(
        "The pooled embeddings contain useful category"
    )

    print(
        "information, but some categories are conflated."
    )

    print(
        "Token-level text representations should still"
    )

    print(
        "be strongly considered for Stage 1."
    )

elif final_accuracy >= 0.40:

    print(
        "INTERPRETATION: WEAK SEMANTIC SEPARATION"
    )

    print(
        "The pooled embeddings do not reliably separate"
    )

    print(
        "the category vocabulary."
    )

    print(
        "Token-level DistilBERT representations should be"
    )

    print(
        "used in the reference grounding architecture."
    )

else:

    print(
        "INTERPRETATION: VERY WEAK SEMANTIC SEPARATION"
    )

    print(
        "The pooled embedding is insufficient for"
    )

    print(
        "fine-grained semantic category discrimination."
    )

    print(
        "Token-level text representations become a"
    )

    print(
        "necessary part of the redesign."
    )


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)