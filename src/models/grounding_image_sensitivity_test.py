import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.grounding_dataset import GroundingDataset
from src.models.grounding_model import GroundingModel


# ============================================================
# Configuration
# ============================================================

NUM_TRAIN_EXAMPLES = 256
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

# We will search for this exact semantic query first.
# The script has fallbacks if it is not present exactly.
TARGET_QUERY = "largest patch of urban fabric"

NUM_TEST_IMAGES = 5


# ============================================================
# Collate
# ============================================================

def collate_fn(batch):
    return {
        "s1": torch.stack([x["s1"] for x in batch]),
        "s2": torch.stack([x["s2"] for x in batch]),
        "text_embedding": torch.stack(
            [x["text_embedding"] for x in batch]
        ),
        "point": torch.stack([x["point"] for x in batch]),
        "target_box": torch.stack(
            [x["target_box"] for x in batch]
        ),
        "mode_id": torch.stack([x["mode_id"] for x in batch]),
    }


# ============================================================
# Helpers
# ============================================================

def clean_query(text):
    """
    Extract the semantic phrase inside <ref>...</ref>.
    """

    text = str(text).strip()

    if "<ref>" in text and "</ref>" in text:

        start = text.find("<ref>") + len("<ref>")
        end = text.find("</ref>", start)

        if end > start:
            return text[start:end].strip()

    return text


def normalize_box(box):
    x1, y1, x2, y2 = [float(v) for v in box]

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    return [x1, y1, x2, y2]


def box_l1(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def box_center(box):
    x1, y1, x2, y2 = box

    return [
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    ]


# ============================================================
# Find reference examples for one exact query
# ============================================================

def find_same_query_examples(dataset, query, num_examples):

    df = dataset.df

    if "input" not in df.columns:
        raise RuntimeError(
            "Expected 'input' column in dataset.df."
        )

    candidates = []

    seen_patches = set()

    for idx in range(len(df)):

        if df.iloc[idx]["category"] != "reference":
            continue

        candidate_query = clean_query(
            df.iloc[idx]["input"]
        )

        if candidate_query != query:
            continue

        item = dataset[idx]

        patch_id = item["patch_id"]

        # We specifically want different images.
        if patch_id in seen_patches:
            continue

        seen_patches.add(patch_id)

        candidates.append(
            {
                "dataset_index": idx,
                "query": candidate_query,
                "patch_id": patch_id,
            }
        )

        if len(candidates) >= num_examples:
            break

    return candidates


# ============================================================
# Fallback: choose a query containing "largest" + "urban"
# ============================================================

def find_fallback_urban_examples(
    dataset,
    num_examples,
):

    df = dataset.df

    candidates = []
    seen_patches = set()

    for idx in range(len(df)):

        if df.iloc[idx]["category"] != "reference":
            continue

        query = clean_query(
            df.iloc[idx]["input"]
        )

        lower = query.lower()

        if "largest" not in lower:
            continue

        if "urban" not in lower:
            continue

        item = dataset[idx]

        patch_id = item["patch_id"]

        if patch_id in seen_patches:
            continue

        seen_patches.add(patch_id)

        candidates.append(
            {
                "dataset_index": idx,
                "query": query,
                "patch_id": patch_id,
            }
        )

        if len(candidates) >= num_examples:
            break

    return candidates


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("GROUNDING IMAGE SENSITIVITY TEST")
print("=" * 70)

dataset = GroundingDataset()

print(
    f"Dataset size: {len(dataset):,}"
)

# ------------------------------------------------------------
# Find reference examples
# ------------------------------------------------------------

reference_indices = [
    i
    for i, category in enumerate(
        dataset.df["category"].tolist()
    )
    if category == "reference"
]

print(
    f"Available reference examples: "
    f"{len(reference_indices):,}"
)

assert len(reference_indices) >= NUM_TRAIN_EXAMPLES


# ============================================================
# Train same current reference model
# ============================================================

train_indices = reference_indices[
    :NUM_TRAIN_EXAMPLES
]

loader = DataLoader(
    Subset(dataset, train_indices),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

model = GroundingModel()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

print()
print(
    f"Training examples: {NUM_TRAIN_EXAMPLES}"
)

print(
    f"Batch size:        {BATCH_SIZE}"
)

print(
    f"Epochs:            {EPOCHS}"
)

print(
    f"Trainable parameters: "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# Train
# ============================================================

print()
print("=" * 70)
print("TRAINING REFERENCE MODEL")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    model.train()

    epoch_loss = 0.0
    total = 0

    for batch in loader:

        predicted = model(
            s1=batch["s1"],
            s2=batch["s2"],
            text_embedding=batch["text_embedding"],
            point=batch["point"],
            mode_id=batch["mode_id"],
        )

        target = batch["target_box"]

        loss = F.smooth_l1_loss(
            predicted,
            target,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = target.size(0)

        epoch_loss += loss.item() * n
        total += n

    epoch_loss /= total

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f}"
    )


# ============================================================
# Find same-query / different-image examples
# ============================================================

print()
print("=" * 70)
print("SEARCHING FOR SAME-QUERY / DIFFERENT-IMAGE EXAMPLES")
print("=" * 70)

tests = find_same_query_examples(
    dataset,
    TARGET_QUERY,
    NUM_TEST_IMAGES,
)

if len(tests) < NUM_TEST_IMAGES:

    print(
        f"Exact query found only {len(tests)} times."
    )

    print(
        "Using fallback search for largest + urban."
    )

    tests = find_fallback_urban_examples(
        dataset,
        NUM_TEST_IMAGES,
    )


if len(tests) < 3:

    print()
    print(
        "ERROR: Could not find at least 3 different images "
        "with the same/similar query."
    )

    print(
        "No meaningful interpretation can be made."
    )

    sys.exit(1)


# ============================================================
# Report selected examples
# ============================================================

print()

for i, test in enumerate(tests):

    print(
        f"{i}: "
        f"row={test['dataset_index']} | "
        f"patch={test['patch_id']} | "
        f"query={test['query']}"
    )


# ============================================================
# Fixed semantic query, varying images
# ============================================================

print()
print("=" * 70)
print("PREDICTIONS — SAME QUERY, DIFFERENT IMAGES")
print("=" * 70)

model.eval()

results = []

with torch.no_grad():

    for test in tests:

        item = dataset[
            test["dataset_index"]
        ]

        # ----------------------------------------------------
        # IMAGE CHANGES
        # ----------------------------------------------------

        s1 = item["s1"].unsqueeze(0)
        s2 = item["s2"].unsqueeze(0)

        point = item["point"].unsqueeze(0)

        # ----------------------------------------------------
        # TEXT STAYS SEMANTICALLY IDENTICAL
        #
        # The selected examples should contain the same
        # reference query. Therefore their text embeddings
        # represent the same query.
        # ----------------------------------------------------

        text_embedding = (
            item["text_embedding"]
            .unsqueeze(0)
        )

        mode_id = torch.tensor(
            [1],
            dtype=torch.long,
        )

        prediction = model(
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
            point=point,
            mode_id=mode_id,
        )

        predicted_box = normalize_box(
            prediction[0].cpu().tolist()
        )

        center = box_center(
            predicted_box
        )

        target_box = normalize_box(
            item["target_box"].tolist()
        )

        results.append(
            {
                "query": test["query"],
                "patch_id": test["patch_id"],
                "predicted_box": predicted_box,
                "target_box": target_box,
                "center": center,
            }
        )

        print()
        print(
            f"Image {len(results)}"
        )

        print(
            f"Patch: {test['patch_id']}"
        )

        print(
            f"Query: {test['query']}"
        )

        print(
            "Predicted: ["
            + ", ".join(
                f"{v:.4f}"
                for v in predicted_box
            )
            + "]"
        )

        print(
            "Target:    ["
            + ", ".join(
                f"{v:.4f}"
                for v in target_box
            )
            + "]"
        )

        print(
            "Pred center: ["
            + ", ".join(
                f"{v:.4f}"
                for v in center
            )
            + "]"
        )


# ============================================================
# Compare predictions across images
# ============================================================

print()
print("=" * 70)
print("IMAGE SENSITIVITY ANALYSIS")
print("=" * 70)

all_differences = []

for i in range(len(results)):

    for j in range(i + 1, len(results)):

        difference = box_l1(
            results[i]["predicted_box"],
            results[j]["predicted_box"],
        )

        all_differences.append(
            difference
        )

        print(
            f"Image {i + 1} vs Image {j + 1}"
            f" → predicted-box L1 = {difference:.6f}"
        )


mean_difference = (
    sum(all_differences)
    / len(all_differences)
)


# ============================================================
# Center movement
# ============================================================

centers = [
    r["center"]
    for r in results
]

center_differences = []

for i in range(len(centers)):

    for j in range(i + 1, len(centers)):

        difference = (
            abs(centers[i][0] - centers[j][0])
            +
            abs(centers[i][1] - centers[j][1])
        )

        center_differences.append(
            difference
        )

mean_center_difference = (
    sum(center_differences)
    / len(center_differences)
)


print()
print(
    f"Mean predicted-box L1 difference: "
    f"{mean_difference:.6f}"
)

print(
    f"Mean predicted-center L1 difference: "
    f"{mean_center_difference:.6f}"
)


# ============================================================
# Interpretation
# ============================================================

print()
print("=" * 70)
print("IMAGE SENSITIVITY RESULT")
print("=" * 70)

if mean_difference < 0.05:

    print(
        "RESULT: VERY LOW IMAGE SENSITIVITY"
    )

    print()
    print(
        "The model produces almost the same box"
    )

    print(
        "for different images under the same query."
    )

    print()
    print(
        "This strongly supports the hypothesis that"
    )

    print(
        "the model has learned approximately"
    )

    print(
        "P(box | query/category)"
    )

    print()
    print(
        "rather than performing true image-conditioned"
    )

    print(
        "semantic grounding."
    )

elif mean_difference < 0.12:

    print(
        "RESULT: LOW IMAGE SENSITIVITY"
    )

    print()
    print(
        "The model changes the prediction somewhat,"
    )

    print(
        "but much less than expected for different images."
    )

elif mean_difference < 0.25:

    print(
        "RESULT: MODERATE IMAGE SENSITIVITY"
    )

    print()
    print(
        "The model is responding to image content,"
    )

    print(
        "but its spatial predictions remain strongly"
    )

    print(
        "biased toward common locations/sizes."
    )

else:

    print(
        "RESULT: HIGH IMAGE SENSITIVITY"
    )

    print()
    print(
        "The predicted box changes substantially"
    )

    print(
        "when the image changes while the query stays fixed."
    )

    print()
    print(
        "This indicates that the model is genuinely"
    )

    print(
        "using image information."
    )


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)