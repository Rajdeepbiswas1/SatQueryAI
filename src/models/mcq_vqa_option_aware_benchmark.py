from pathlib import Path
import random
import sys
import time

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from models.mcq_vqa_option_aware import MCQOptionAwareVQA


# ============================================================
# PATHS
# ============================================================

SPLIT_FILE = Path(
    r"data\processed\mcq_vqa_split.parquet"
)

MAPPING_FILE = Path(
    r"data\processed\mcq_question_embeddings"
    r"\mcq_row_mapping.parquet"
)

OPTION_CACHE = Path(
    r"data\processed\mcq_option_embeddings"
)

DOFA_CACHE = Path(
    r"data\processed\dofa_features_download"
    r"\dofa_features"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32
BENCHMARK_BATCHES = 50

LEARNING_RATE = 1e-4

DEVICE = torch.device("cpu")


# ============================================================
# LOAD OPTION EMBEDDINGS
# ============================================================

print("Loading option-aware embeddings...")

option_embeddings = {}

for file in sorted(
    OPTION_CACHE.glob("shard_*.pt")
):

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    indices = data[
        "question_indices"
    ].tolist()

    question = data[
        "question_embeddings"
    ].float()

    option_a = data[
        "option_a_embeddings"
    ].float()

    option_b = data[
        "option_b_embeddings"
    ].float()

    option_c = data[
        "option_c_embeddings"
    ].float()

    option_d = data[
        "option_d_embeddings"
    ].float()

    for i, index in enumerate(indices):

        option_embeddings[int(index)] = {
            "question": question[i],
            "a": option_a[i],
            "b": option_b[i],
            "c": option_c[i],
            "d": option_d[i],
        }


print(
    "Cached MCQ questions:",
    len(option_embeddings)
)


# ============================================================
# LOAD SPLIT + MAPPING
# ============================================================

print("\nLoading MCQ data...")

split = pd.read_parquet(
    SPLIT_FILE
)

mapping = pd.read_parquet(
    MAPPING_FILE
)

mapping_lookup = {
    str(row["ID"]): int(
        row["question_embedding_index"]
    )
    for _, row in mapping.iterrows()
}

mcq = split[
    split["type"] == "mcq"
].copy()

mcq["question_embedding_index"] = (
    mcq["ID"]
    .astype(str)
    .map(mapping_lookup)
)

train = mcq[
    mcq["model_split"] == "train"
].copy()

print(
    "Training rows:",
    len(train)
)


# ============================================================
# PREPARE SAMPLES
# ============================================================

label_map = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
}

samples = []

for _, row in train.iterrows():

    answer = (
        str(row["output"])
        .strip()
        .lower()
    )

    if answer not in label_map:
        raise RuntimeError(
            f"Unexpected answer: {answer}"
        )

    samples.append(
        {
            "patch_id": str(
                row["patch_id"]
            ),
            "embedding_index": int(
                row["question_embedding_index"]
            ),
            "label": label_map[answer],
        }
    )


# ============================================================
# INDEX DOFA SHARDS
# ============================================================

print("\nIndexing DOFA shards...")

dofa_files = sorted(
    DOFA_CACHE.glob("shard_*.pt")
)

patch_to_shard = {}
patch_to_position = {}

for shard_index, file in enumerate(
    dofa_files
):

    data = torch.load(
        file,
        map_location="cpu",
        weights_only=False,
    )

    for position, patch_id in enumerate(
        data["patch_ids"]
    ):

        patch_id = str(patch_id)

        patch_to_shard[
            patch_id
        ] = shard_index

        patch_to_position[
            patch_id
        ] = position

print(
    "DOFA shards:",
    len(dofa_files)
)


# ============================================================
# GROUP SAMPLES BY SHARD
# ============================================================

groups = {
    shard: []
    for shard in range(
        len(dofa_files)
    )
}

for index, sample in enumerate(
    samples
):

    patch_id = sample[
        "patch_id"
    ]

    shard = patch_to_shard[
        patch_id
    ]

    groups[shard].append(
        index
    )


# ============================================================
# MODEL
# ============================================================

model = MCQOptionAwareVQA().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

criterion = nn.CrossEntropyLoss()


print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# ============================================================
# WARMUP
# ============================================================

print("\nRunning warm-up...")

first_shard = next(
    shard
    for shard, indices
    in groups.items()
    if indices
)

data = torch.load(
    dofa_files[first_shard],
    map_location="cpu",
    weights_only=False,
)

warmup_indices = groups[
    first_shard
][:BATCH_SIZE]

s1_list = []
s2_list = []
question_list = []
options_list = []
labels_list = []

for sample_index in warmup_indices:

    sample = samples[
        sample_index
    ]

    patch_id = sample[
        "patch_id"
    ]

    embedding_index = sample[
        "embedding_index"
    ]

    position = patch_to_position[
        patch_id
    ]

    cached = option_embeddings[
        embedding_index
    ]

    s1_list.append(
        data["s1"][
            position
        ].float()
    )

    s2_list.append(
        data["s2"][
            position
        ].float()
    )

    question_list.append(
        cached["question"]
    )

    options_list.append(
        torch.stack(
            [
                cached["a"],
                cached["b"],
                cached["c"],
                cached["d"],
            ]
        )
    )

    labels_list.append(
        sample["label"]
    )

s1 = torch.stack(s1_list)
s2 = torch.stack(s2_list)
question = torch.stack(question_list)
options = torch.stack(options_list)
labels = torch.tensor(
    labels_list,
    dtype=torch.long,
)

optimizer.zero_grad(
    set_to_none=True
)

logits = model(
    s1,
    s2,
    question,
    options,
)

loss = criterion(
    logits,
    labels,
)

loss.backward()
optimizer.step()

del data


# ============================================================
# BENCHMARK
# ============================================================

print(
    f"\nStarting "
    f"{BENCHMARK_BATCHES}-batch benchmark..."
)

# Use shard order rather than jumping randomly
# between shards.
shard_order = [
    shard
    for shard, indices
    in groups.items()
    if indices
]

random.Random(42).shuffle(
    shard_order
)

start_time = time.perf_counter()

batches_done = 0

for shard in shard_order:

    if batches_done >= BENCHMARK_BATCHES:
        break

    data = torch.load(
        dofa_files[shard],
        map_location="cpu",
        weights_only=False,
    )

    indices = list(
        groups[shard]
    )

    random.Random(
        42 + shard
    ).shuffle(indices)

    for start in range(
        0,
        len(indices),
        BATCH_SIZE,
    ):

        if batches_done >= BENCHMARK_BATCHES:
            break

        batch_indices = indices[
            start:start + BATCH_SIZE
        ]

        s1_list = []
        s2_list = []
        question_list = []
        options_list = []
        labels_list = []

        for sample_index in batch_indices:

            sample = samples[
                sample_index
            ]

            patch_id = sample[
                "patch_id"
            ]

            embedding_index = sample[
                "embedding_index"
            ]

            position = patch_to_position[
                patch_id
            ]

            cached = option_embeddings[
                embedding_index
            ]

            s1_list.append(
                data["s1"][
                    position
                ].float()
            )

            s2_list.append(
                data["s2"][
                    position
                ].float()
            )

            question_list.append(
                cached["question"]
            )

            options_list.append(
                torch.stack(
                    [
                        cached["a"],
                        cached["b"],
                        cached["c"],
                        cached["d"],
                    ]
                )
            )

            labels_list.append(
                sample["label"]
            )

        s1 = torch.stack(s1_list)
        s2 = torch.stack(s2_list)
        question = torch.stack(
            question_list
        )

        options = torch.stack(
            options_list
        )

        labels = torch.tensor(
            labels_list,
            dtype=torch.long,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            s1,
            s2,
            question,
            options,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        batches_done += 1

        if batches_done % 10 == 0:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            avg = (
                elapsed
                / batches_done
            )

            print(
                f"Batch "
                f"{batches_done:3d}/"
                f"{BENCHMARK_BATCHES} | "
                f"Elapsed "
                f"{elapsed:.1f}s | "
                f"Avg "
                f"{avg:.3f}s/batch"
            )

    del data


# ============================================================
# ESTIMATES
# ============================================================

elapsed = (
    time.perf_counter()
    - start_time
)

avg_batch_time = (
    elapsed
    / BENCHMARK_BATCHES
)

batches_per_epoch = (
    len(samples)
    + BATCH_SIZE
    - 1
) // BATCH_SIZE

epoch_seconds = (
    avg_batch_time
    * batches_per_epoch
)

ten_epoch_seconds = (
    epoch_seconds
    * 10
)


print("\n" + "=" * 60)
print("OPTION-AWARE MCQ SPEED BENCHMARK")
print("=" * 60)

print(
    f"Actual batches:       "
    f"{batches_done}"
)

print(
    f"Benchmark time:       "
    f"{elapsed:.2f} sec"
)

print(
    f"Average batch time:   "
    f"{avg_batch_time:.3f} sec"
)

print(
    f"Batches per epoch:    "
    f"{batches_per_epoch}"
)

print(
    f"Estimated 1 epoch:    "
    f"{epoch_seconds / 60:.2f} min"
)

print(
    f"Estimated 10 epochs:  "
    f"{ten_epoch_seconds / 3600:.2f} hr"
)

print("=" * 60)
print("BENCHMARK COMPLETE")