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

NUM_EXAMPLES = 256
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3


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


def clean_query(text):
    """
    Extract the text inside <ref>...</ref> when present.
    """

    text = str(text).strip()

    if "<ref>" in text and "</ref>" in text:

        start = text.find("<ref>") + len("<ref>")
        end = text.find("</ref>", start)

        if end > start:
            return text[start:end].strip()

    return text


# ============================================================
# Find controlled "largest + category" queries
# ============================================================

def select_controlled_queries(dataset, reference_indices):

    df = dataset.df

    text_column = "input"

    # --------------------------------------------------------
    # We want the superlative to stay constant:
    #
    # largest + [category]
    #
    # We search the actual reference annotations and choose
    # phrases containing "largest" plus the desired category.
    # --------------------------------------------------------

    category_specs = [
        ("FOREST", ["forest"]),
        ("URBAN", ["urban fabric", "urban"]),
        ("WATER", ["water"]),
        ("PASTURES", ["pasture", "pastures"]),
        ("CROPS", ["permanent crops", "crop", "crops"]),
    ]

    selected = []

    for label, keywords in category_specs:

        candidates = []

        for idx in reference_indices:

            raw_text = df.iloc[idx][text_column]

            query = clean_query(raw_text)

            lower = query.lower()

            # Must contain the word "largest".
            if "largest" not in lower:
                continue

            # Must contain the desired semantic category.
            if not any(
                keyword in lower
                for keyword in keywords
            ):
                continue

            candidates.append(
                {
                    "dataset_index": idx,
                    "query": query,
                }
            )

        if not candidates:
            print(
                f"[NOT FOUND] No 'largest' query found "
                f"for {label}"
            )
            continue

        # Prefer the shortest matching phrase because it minimizes
        # variation in wording beyond the category itself.
        candidates.sort(
            key=lambda x: len(x["query"])
        )

        selected.append(
            {
                "label": label,
                "dataset_index": candidates[0]["dataset_index"],
                "query": candidates[0]["query"],
            }
        )

    return selected


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("CONTROLLED LEXICAL-SHORTCUT GROUNDING TEST")
print("=" * 70)

dataset = GroundingDataset()

print(f"Dataset size: {len(dataset):,}")

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

assert len(reference_indices) >= NUM_EXAMPLES


# ============================================================
# Train exactly the same 256-example reference model
# ============================================================

reference_subset = reference_indices[:NUM_EXAMPLES]

loader = DataLoader(
    Subset(dataset, reference_subset),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

print(f"Training examples: {NUM_EXAMPLES}")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Epochs:            {EPOCHS}")

model = GroundingModel()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

print(
    f"Trainable parameters: "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)

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
# Fixed image
# ============================================================

print()
print("=" * 70)
print("FIXED IMAGE")
print("=" * 70)

fixed_idx = reference_subset[0]

fixed_item = dataset[fixed_idx]

fixed_s1 = fixed_item["s1"].unsqueeze(0)
fixed_s2 = fixed_item["s2"].unsqueeze(0)
fixed_point = fixed_item["point"].unsqueeze(0)

fixed_mode = torch.tensor(
    [1],
    dtype=torch.long,
)

print(f"Dataset index: {fixed_idx}")
print(f"Patch ID:      {fixed_item['patch_id']}")
print(f"S1 shape:      {tuple(fixed_s1.shape)}")
print(f"S2 shape:      {tuple(fixed_s2.shape)}")


# ============================================================
# Controlled queries
# ============================================================

tests = select_controlled_queries(
    dataset,
    reference_indices,
)

print()
print("=" * 70)
print("CONTROLLED TEST QUERIES")
print("=" * 70)

for test in tests:

    print(
        f"{test['label']:10s} | "
        f"row={test['dataset_index']} | "
        f"{test['query']}"
    )

if len(tests) < 3:

    print()
    print(
        "ERROR: Fewer than 3 controlled categories were found."
    )

    print(
        "Do not interpret the test."
    )

    sys.exit(1)


# ============================================================
# Run predictions
# ============================================================

print()
print("=" * 70)
print("PREDICTIONS — SAME IMAGE, SAME 'LARGEST'")
print("=" * 70)

model.eval()

results = []

with torch.no_grad():

    for test in tests:

        item = dataset[
            test["dataset_index"]
        ]

        text_embedding = (
            item["text_embedding"]
            .unsqueeze(0)
        )

        prediction = model(
            s1=fixed_s1,
            s2=fixed_s2,
            text_embedding=text_embedding,
            point=fixed_point,
            mode_id=fixed_mode,
        )

        box = normalize_box(
            prediction[0].cpu().tolist()
        )

        center = box_center(box)

        results.append(
            {
                "label": test["label"],
                "query": test["query"],
                "box": box,
                "center": center,
            }
        )

        print()
        print(
            f"{test['label']}: "
            f"{test['query']}"
        )

        print(
            "Box: ["
            + ", ".join(
                f"{v:.4f}"
                for v in box
            )
            + "]"
        )

        print(
            "Center: ["
            + ", ".join(
                f"{v:.4f}"
                for v in center
            )
            + "]"
        )


# ============================================================
# Pairwise differences
# ============================================================

print()
print("=" * 70)
print("PAIRWISE BOX DIFFERENCES")
print("=" * 70)

all_differences = []

for i in range(len(results)):

    for j in range(i + 1, len(results)):

        difference = box_l1(
            results[i]["box"],
            results[j]["box"],
        )

        all_differences.append(
            difference
        )

        print(
            f"{results[i]['label']:10s} vs "
            f"{results[j]['label']:10s} "
            f"→ L1 = {difference:.6f}"
        )

mean_difference = (
    sum(all_differences)
    / len(all_differences)
)


# ============================================================
# Interpretation
# ============================================================

print()
print("=" * 70)
print("CONTROLLED TEST RESULT")
print("=" * 70)

print(
    f"Mean pairwise box L1 difference: "
    f"{mean_difference:.6f}"
)

if mean_difference < 0.05:

    print()
    print(
        "RESULT: CATEGORIES ARE CLUSTERING STRONGLY."
    )

    print(
        "Despite holding 'largest' constant, the model"
    )

    print(
        "produces nearly identical boxes."
    )

    print(
        "This strongly supports a lexical-shortcut/"
    )

    print(
        "weak category-to-region binding hypothesis."
    )

elif mean_difference < 0.15:

    print()
    print(
        "RESULT: WEAK CATEGORY SEPARATION."
    )

    print(
        "The model responds somewhat to category changes,"
    )

    print(
        "but different categories still produce fairly"
    )

    print(
        "similar spatial predictions."
    )

else:

    print()
    print(
        "RESULT: STRONG CATEGORY SENSITIVITY."
    )

    print(
        "Changing only the category produces substantially"
    )

    print(
        "different predicted regions."
    )

    print(
        "The lexical-shortcut hypothesis is therefore"
    )

    print(
        "not sufficient by itself to explain the failure."
    )

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)