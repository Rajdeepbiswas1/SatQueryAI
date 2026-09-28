from pathlib import Path
import sys
import torch
import torch.nn as nn
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS = PROJECT_ROOT / "data" / "processed" / "binary_vqa_split.parquet"
QUESTION_CACHE = PROJECT_ROOT / "data" / "processed" / "binary_question_embeddings"
DOFA_CACHE = PROJECT_ROOT / "data" / "processed" / "dofa_features_download" / "dofa_features"
CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "binary_vqa" / "best.pt"

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from models.feature_fusion import CrossModalFusion


# ============================================================
# MODEL — EXACT CACHED BINARY VQA ARCHITECTURE
# ============================================================

class CachedBinaryVQA(nn.Module):

    def __init__(self):
        super().__init__()

        self.visual_fusion = CrossModalFusion(
            input_dim=768,
            hidden_dim=256,
        )

        self.question_projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )

        self.visual_key = nn.Linear(256, 256)

        self.question_query = nn.Linear(256, 256)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, s1, s2, question):

        visual = self.visual_fusion(s1, s2)

        question = self.question_projection(question)

        keys = self.visual_key(visual)
        query = self.question_query(question)

        scores = torch.bmm(
            keys,
            query.unsqueeze(-1),
        ).squeeze(-1)

        scores = scores / (256 ** 0.5)

        attention = torch.softmax(scores, dim=1)

        visual_summary = torch.bmm(
            attention.unsqueeze(1),
            visual,
        ).squeeze(1)

        interaction = visual_summary * question

        combined = torch.cat(
            [
                visual_summary,
                question,
                interaction,
            ],
            dim=1,
        )

        return self.classifier(combined)


# ============================================================
# CHECK PATHS
# ============================================================

print("=" * 72)
print("SATQUERYAI — FROZEN BINARY VQA CHECKPOINT VERIFICATION")
print("=" * 72)

for name, path in [
    ("Annotations", ANNOTATIONS),
    ("Question cache", QUESTION_CACHE),
    ("DOFA cache", DOFA_CACHE),
    ("Checkpoint", CHECKPOINT),
]:
    status = "PASS" if path.exists() else "FAIL"
    print(f"{name:20s}: {status} -> {path}")

if not ANNOTATIONS.exists():
    raise FileNotFoundError(ANNOTATIONS)

if not QUESTION_CACHE.exists():
    raise FileNotFoundError(QUESTION_CACHE)

if not DOFA_CACHE.exists():
    raise FileNotFoundError(DOFA_CACHE)

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)


# ============================================================
# LOAD QUESTION EMBEDDINGS
# ============================================================

print("\nLoading question embeddings...")

question_ids = []
question_embeddings = []

question_files = sorted(
    QUESTION_CACHE.glob("shard_*.pt")
)

if not question_files:
    raise RuntimeError("No question embedding shards found.")

for file in question_files:

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    question_ids.extend(
        str(x) for x in data["ids"]
    )

    question_embeddings.append(
        data["question_embeddings"].float()
    )

question_embeddings = torch.cat(
    question_embeddings,
    dim=0,
)

question_lookup = {
    qid: question_embeddings[i]
    for i, qid in enumerate(question_ids)
}

print("Question embedding shape:", tuple(question_embeddings.shape))
print("Question lookup size:", len(question_lookup))


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading annotations...")

annotations = pd.read_parquet(ANNOTATIONS)

val_df = annotations[
    annotations["model_split"] == "val"
].copy()

val_df["answer_clean"] = (
    val_df["output"]
    .astype(str)
    .str.strip()
    .str.lower()
)

val_df = val_df[
    val_df["answer_clean"].isin(["yes", "no"])
].copy()

if len(val_df) == 0:
    raise RuntimeError("No binary validation samples found.")

print("Binary validation rows:", len(val_df))

# Use a small real validation batch.
samples_df = val_df.head(8).copy()

print("\nSelected validation samples:")

for i, row in samples_df.iterrows():
    print(
        f"  ID={row['ID']} | "
        f"patch={row['patch_id']} | "
        f"answer={row['answer_clean']}"
    )


# ============================================================
# INDEX DOFA SHARDS
# ============================================================

print("\nIndexing DOFA shards...")

dofa_files = sorted(
    DOFA_CACHE.glob("shard_*.pt")
)

if not dofa_files:
    raise RuntimeError("No DOFA shards found.")

patch_to_shard = {}
patch_to_position = {}

for shard_index, file in enumerate(dofa_files):

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    for position, patch_id in enumerate(data["patch_ids"]):

        patch_id = str(patch_id)

        patch_to_shard[patch_id] = shard_index
        patch_to_position[patch_id] = position

    del data

print("DOFA shards:", len(dofa_files))
print("Indexed patches:", len(patch_to_shard))


# ============================================================
# COLLECT REQUIRED PATCHES
# ============================================================

required = []

for _, row in samples_df.iterrows():

    qid = str(row["ID"])
    patch_id = str(row["patch_id"])

    if qid not in question_lookup:
        raise RuntimeError(
            f"Missing question embedding for ID: {qid}"
        )

    if patch_id not in patch_to_shard:
        raise RuntimeError(
            f"Missing DOFA patch: {patch_id}"
        )

    required.append(
        {
            "id": qid,
            "patch_id": patch_id,
            "question": question_lookup[qid],
            "label": 1 if row["answer_clean"] == "yes" else 0,
        }
    )


# ============================================================
# LOAD ONLY REQUIRED DOFA SHARDS
# ============================================================

print("\nLoading required DOFA features...")

required_by_shard = {}

for sample_index, sample in enumerate(required):

    shard = patch_to_shard[sample["patch_id"]]

    required_by_shard.setdefault(
        shard,
        []
    ).append(sample_index)

s1_list = [None] * len(required)
s2_list = [None] * len(required)

for shard, sample_indices in required_by_shard.items():

    file = dofa_files[shard]

    print(
        f"  Loading shard {shard + 1}/{len(dofa_files)}"
    )

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    for sample_index in sample_indices:

        sample = required[sample_index]

        position = patch_to_position[
            sample["patch_id"]
        ]

        s1_list[sample_index] = (
            data["s1"][position]
            .float()
            .clone()
        )

        s2_list[sample_index] = (
            data["s2"][position]
            .float()
            .clone()
        )

    del data


s1 = torch.stack(s1_list)
s2 = torch.stack(s2_list)

question = torch.stack(
    [sample["question"] for sample in required]
)

labels = torch.tensor(
    [sample["label"] for sample in required],
    dtype=torch.long,
)


# ============================================================
# SHAPE / FINITENESS CHECKS
# ============================================================

print("\nInput shapes:")

print("  S1:", tuple(s1.shape))
print("  S2:", tuple(s2.shape))
print("  Q :", tuple(question.shape))

assert s1.shape == (len(required), 196, 768)
assert s2.shape == (len(required), 196, 768)
assert question.shape == (len(required), 768)

assert torch.isfinite(s1).all()
assert torch.isfinite(s2).all()
assert torch.isfinite(question).all()

print("Input shape/finiteness checks: PASS")


# ============================================================
# LOAD FROZEN CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

model = CachedBinaryVQA()

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

if "model_state_dict" not in checkpoint:
    raise RuntimeError(
        "Checkpoint does not contain model_state_dict."
    )

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

param_count = sum(
    p.numel()
    for p in model.parameters()
)

print("Model parameters:", param_count)
print("Checkpoint epoch:", checkpoint.get("epoch"))
print(
    "Recorded validation accuracy:",
    checkpoint.get("val_accuracy")
)


# ============================================================
# FROZEN INFERENCE
# ============================================================

print("\nRunning frozen inference...")

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
# VERIFY OUTPUT
# ============================================================

print("\nOutput shape:", tuple(logits.shape))

assert logits.shape == (len(required), 2)
assert probabilities.shape == (len(required), 2)

assert torch.isfinite(logits).all()
assert torch.isfinite(probabilities).all()

print("Output shape/finiteness checks: PASS")


# ============================================================
# REPORT RESULTS
# ============================================================

correct = 0

print("\nValidation predictions:")

for i, sample in enumerate(required):

    predicted_label = int(
        predictions[i].item()
    )

    predicted_text = (
        "yes"
        if predicted_label == 1
        else "no"
    )

    actual_text = (
        "yes"
        if sample["label"] == 1
        else "no"
    )

    confidence = float(
        probabilities[i, predicted_label].item()
    )

    if predicted_label == sample["label"]:
        correct += 1
        result = "CORRECT"
    else:
        result = "INCORRECT"

    print(
        f"  {i + 1}. "
        f"ID={sample['id']} | "
        f"pred={predicted_text} | "
        f"actual={actual_text} | "
        f"confidence={confidence:.4f} | "
        f"{result}"
    )


accuracy = correct / len(required)

print("\n" + "=" * 72)
print("FROZEN BINARY VQA VERIFICATION COMPLETE")
print("=" * 72)

print(f"Samples tested : {len(required)}")
print(f"Correct        : {correct}")
print(f"Accuracy       : {accuracy:.4f}")

print("\nBINARY VQA CHECKPOINT FORWARD PASS: PASS")
print("=" * 72)