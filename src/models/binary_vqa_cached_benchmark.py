from pathlib import Path
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "src")

from models.feature_fusion import CrossModalFusion


ANNOTATIONS = Path(r"data\processed\binary_vqa_split.parquet")
QUESTION_CACHE = Path(r"data\processed\binary_question_embeddings")
DOFA_CACHE = Path(r"data\processed\dofa_features_download\dofa_features")

BATCH_SIZE = 8
BENCHMARK_BATCHES = 50
DEVICE = torch.device("cpu")


# ============================================================
# QUESTION EMBEDDINGS
# ============================================================

print("Loading question embeddings...")

question_ids = []
question_embeddings = []

for file in sorted(QUESTION_CACHE.glob("shard_*.pt")):
    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    question_ids.extend(data["ids"])
    question_embeddings.append(
        data["question_embeddings"].float()
    )

question_embeddings = torch.cat(
    question_embeddings,
    dim=0,
)

question_lookup = {
    str(qid): question_embeddings[i]
    for i, qid in enumerate(question_ids)
}

print("Question embeddings:", tuple(question_embeddings.shape))


# ============================================================
# ANNOTATIONS
# ============================================================

annotations = pd.read_parquet(ANNOTATIONS)

train_df = annotations[
    annotations["model_split"] == "train"
].copy()

print("Training rows:", len(train_df))


samples = []

for _, row in train_df.iterrows():

    answer = str(row["output"]).strip().lower()

    if answer not in ("yes", "no"):
        continue

    qid = str(row["ID"])
    patch_id = str(row["patch_id"])

    if qid not in question_lookup:
        raise RuntimeError(
            f"Missing question embedding: {qid}"
        )

    samples.append(
        (
            patch_id,
            qid,
            1 if answer == "yes" else 0,
        )
    )

print("Binary samples:", len(samples))


# ============================================================
# DOFA SHARD INDEX
# ============================================================

print("\nIndexing DOFA shards...")

dofa_files = sorted(DOFA_CACHE.glob("shard_*.pt"))

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

print("DOFA shards:", len(dofa_files))
print("Indexed patches:", len(patch_to_shard))


# ============================================================
# SHARD-AWARE DATASET
# ============================================================

class CachedBinaryDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

        self.current_shard_index = None
        self.current_shard = None

    def __len__(self):
        return len(self.samples)

    def _load_shard(self, shard_index):

        if self.current_shard_index == shard_index:
            return

        file = dofa_files[shard_index]

        print(
            f"Loading DOFA shard {shard_index:02d}: "
            f"{file.name}"
        )

        self.current_shard = torch.load(
            file,
            map_location="cpu",
            weights_only=False,
        )

        self.current_shard_index = shard_index

    def __getitem__(self, idx):

        patch_id, qid, label = self.samples[idx]

        shard_index = patch_to_shard[patch_id]
        position = patch_to_position[patch_id]

        self._load_shard(shard_index)

        s1 = self.current_shard["s1"][position].float()
        s2 = self.current_shard["s2"][position].float()

        question = question_lookup[qid]

        return (
            s1,
            s2,
            question,
            torch.tensor(label, dtype=torch.long),
        )


dataset = CachedBinaryDataset(samples)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# ============================================================
# MODEL
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

    def forward(
        self,
        s1,
        s2,
        question,
    ):

        visual = self.visual_fusion(
            s1,
            s2,
        )

        question = self.question_projection(
            question
        )

        keys = self.visual_key(visual)
        query = self.question_query(question)

        scores = torch.bmm(
            keys,
            query.unsqueeze(-1),
        ).squeeze(-1)

        scores = scores / (256 ** 0.5)

        attention = torch.softmax(
            scores,
            dim=1,
        )

        visual_summary = torch.bmm(
            attention.unsqueeze(1),
            visual,
        ).squeeze(1)

        interaction = (
            visual_summary * question
        )

        combined = torch.cat(
            [
                visual_summary,
                question,
                interaction,
            ],
            dim=1,
        )

        return self.classifier(combined)


model = CachedBinaryVQA().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

criterion = nn.CrossEntropyLoss()


print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ),
)


# ============================================================
# WARMUP
# ============================================================

print("\nRunning warm-up...")

model.train()

warmup_batch = next(iter(loader))

s1, s2, question, labels = warmup_batch

optimizer.zero_grad(set_to_none=True)

logits = model(
    s1,
    s2,
    question,
)

loss = criterion(
    logits,
    labels,
)

loss.backward()
optimizer.step()


# ============================================================
# BENCHMARK
# ============================================================

print("\nStarting 100-batch benchmark...")

start_time = time.perf_counter()

for batch_number, batch in enumerate(
    loader,
    start=1,
):

    if batch_number > BENCHMARK_BATCHES:
        break

    s1, s2, question, labels = batch

    optimizer.zero_grad(set_to_none=True)

    logits = model(
        s1,
        s2,
        question,
    )

    loss = criterion(
        logits,
        labels,
    )

    loss.backward()
    optimizer.step()

    if batch_number % 10 == 0:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        avg = elapsed / batch_number

        print(
            f"Batch {batch_number:3d}/100 | "
            f"Elapsed {elapsed:.1f}s | "
            f"Avg {avg:.2f}s/batch"
        )


elapsed = (
    time.perf_counter()
    - start_time
)

avg_batch_time = (
    elapsed / BENCHMARK_BATCHES
)

batches_per_epoch = (
    len(samples) + BATCH_SIZE - 1
) // BATCH_SIZE

epoch_seconds = (
    avg_batch_time
    * batches_per_epoch
)

ten_epoch_seconds = (
    epoch_seconds * 10
)

print("\n" + "=" * 60)
print("BINARY VQA TRAINING SPEED BENCHMARK")
print("=" * 60)

print(
    f"100 batches time:    {elapsed:.2f} sec"
)

print(
    f"Average batch time:  "
    f"{avg_batch_time:.3f} sec"
)

print(
    f"Batches per epoch:   "
    f"{batches_per_epoch}"
)

print(
    f"Estimated 1 epoch:   "
    f"{epoch_seconds / 60:.2f} min"
)

print(
    f"Estimated 10 epochs: "
    f"{ten_epoch_seconds / 3600:.2f} hr"
)

print("=" * 60)
print("BENCHMARK COMPLETE")