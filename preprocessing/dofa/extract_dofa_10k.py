import os
import glob
import time

import torch
import torch.nn.functional as F

from torchgeo.models import (
    dofa_base_patch16_224,
    DOFABase16_Weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = (
    "/kaggle/input/datasets/"
    "rajdeepbiswas1/"
    "satqueryai-dofa-inputs-10k/"
    "dofa_inputs_10k"
)

S1_STATS_PATH = (
    "/kaggle/input/datasets/"
    "rajdeepbiswas1/"
    "satqueryai-dofa-inputs-10k/"
    "dofa_s1_stats.pt"
)

S2_STATS_PATH = (
    "/kaggle/input/datasets/"
    "rajdeepbiswas1/"
    "satqueryai-dofa-inputs-10k/"
    "dofa_s2_stats.pt"
)

OUTPUT_DIR = "/kaggle/working/dofa_features"

# Safe starting batch size based on our successful GPU test.
BATCH_SIZE = 8

# Each output file contains at most 256 samples.
SHARD_SIZE = 256


# ============================================================
# DOFA CONFIGURATION
# ============================================================

# Existing tensor channel order:
#
# 0  VH
# 1  VV
# 2  B01
# 3  B02
# 4  B03
# 5  B04
# 6  B05
# 7  B06
# 8  B07
# 9  B08
# 10 B09
# 11 B11
# 12 B12
# 13 B8A

S1_INDICES = [0, 1]

S2_INDICES = [
    5,   # B04
    4,   # B03
    3,   # B02
    6,   # B05
    7,   # B06
    8,   # B07
    9,   # B08
    11,  # B11
    12,  # B12
]

S1_WAVELENGTHS = [
    3.75,
    3.75,
]

S2_WAVELENGTHS = [
    0.665,
    0.560,
    0.490,
    0.705,
    0.740,
    0.783,
    0.842,
    1.610,
    2.190,
]


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type != "cuda":
    raise RuntimeError(
        "CUDA GPU is not available. "
        "This extraction should run on the Kaggle GPU."
    )

print("=" * 70)
print("SATQUERYAI — FULL DOFA FEATURE EXTRACTION")
print("=" * 70)

print("\nPyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(
        f"GPU {i}:",
        torch.cuda.get_device_name(i)
    )

print("\nInput directory:")
print(INPUT_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# FIND INPUT FILES
# ============================================================

paths = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*.pt"
        )
    )
)

print("\nFound input files:", len(paths))

if len(paths) != 10000:
    raise RuntimeError(
        f"Expected exactly 10,000 .pt files, "
        f"found {len(paths)}."
    )


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

print("\nLoading normalization statistics...")

s1_stats = torch.load(
    S1_STATS_PATH,
    map_location="cpu"
)

s2_stats = torch.load(
    S2_STATS_PATH,
    map_location="cpu"
)

s1_mean = s1_stats["mean"].float().to(device)
s1_std = s1_stats["std"].float().to(device)

s2_mean = s2_stats["mean"].float().to(device)
s2_std = s2_stats["std"].float().to(device)

print("S1 mean:", s1_mean)
print("S1 std :", s1_std)

print("S2 mean:", s2_mean)
print("S2 std :", s2_std)


# ============================================================
# LOAD DOFA
# ============================================================

print("\nLoading pretrained DOFA...")

model = dofa_base_patch16_224(
    weights=DOFABase16_Weights.DOFA_MAE
).to(device)

model.eval()

print("DOFA loaded successfully.")


# ============================================================
# HELPER FUNCTION
# ============================================================

@torch.inference_mode()
def extract_spatial_features(
    images,
    indices,
    mean,
    std,
    wavelengths,
):
    """
    Input:
        images: [B, 14, 120, 120]

    Output:
        [B, 196, 768] float16
    """

    # --------------------------------------------------------
    # Select channels
    # --------------------------------------------------------

    x = images[:, indices, :, :]

    # --------------------------------------------------------
    # Channel-wise normalization
    # --------------------------------------------------------

    mean = mean.view(
        1,
        -1,
        1,
        1,
    )

    std = std.view(
        1,
        -1,
        1,
        1,
    )

    x = (x - mean) / std

    # --------------------------------------------------------
    # Convert to DOFA input size
    # --------------------------------------------------------

    x = F.interpolate(
        x,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )

    # --------------------------------------------------------
    # Wavelength tensor
    # --------------------------------------------------------

    wavelengths = torch.tensor(
        wavelengths,
        dtype=x.dtype,
        device=x.device,
    )

    # --------------------------------------------------------
    # DOFA dynamic patch embedding
    # --------------------------------------------------------

    tokens, _ = model.patch_embed(
        x,
        wavelengths,
    )

    # Expected:
    #
    # [B, 196, 768]

    # --------------------------------------------------------
    # Positional embeddings
    # --------------------------------------------------------

    tokens = (
        tokens
        + model.pos_embed[:, 1:, :]
    )

    # --------------------------------------------------------
    # CLS token
    # --------------------------------------------------------

    cls = (
        model.cls_token
        + model.pos_embed[:, :1, :]
    )

    cls = cls.expand(
        tokens.shape[0],
        -1,
        -1,
    )

    tokens = torch.cat(
        [cls, tokens],
        dim=1,
    )

    # Expected:
    #
    # [B, 197, 768]

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    for block in model.blocks:
        tokens = block(tokens)

    # --------------------------------------------------------
    # Remove CLS token
    # --------------------------------------------------------

    spatial_tokens = tokens[:, 1:, :]

    # --------------------------------------------------------
    # Store as FP16
    # --------------------------------------------------------

    spatial_tokens = spatial_tokens.half()

    return spatial_tokens


# ============================================================
# EXTRACTION STATE
# ============================================================

total_samples = len(paths)

processed = 0
shard_id = 0

shard_s1 = []
shard_s2 = []
shard_patch_ids = []

start_time = time.time()


# ============================================================
# MAIN EXTRACTION LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING EXTRACTION")
print("=" * 70)

for start in range(
    0,
    total_samples,
    BATCH_SIZE,
):

    batch_paths = paths[
        start:start + BATCH_SIZE
    ]

    images = []
    patch_ids = []

    # --------------------------------------------------------
    # Load batch from Kaggle input
    # --------------------------------------------------------

    for path in batch_paths:

        data = torch.load(
            path,
            map_location="cpu"
        )

        image = data["image"].float()

        if image.shape != (
            14,
            120,
            120,
        ):
            raise RuntimeError(
                f"Unexpected tensor shape "
                f"{tuple(image.shape)} "
                f"in {path}"
            )

        images.append(image)

        patch_ids.append(
            data["patch_id"]
        )

    images = torch.stack(
        images,
        dim=0,
    )

    # --------------------------------------------------------
    # Move to GPU
    # --------------------------------------------------------

    images = images.to(
        device,
        non_blocking=True,
    )

    # --------------------------------------------------------
    # S1 DOFA
    # --------------------------------------------------------

    s1_features = extract_spatial_features(
        images,
        S1_INDICES,
        s1_mean,
        s1_std,
        S1_WAVELENGTHS,
    )

    # --------------------------------------------------------
    # S2 DOFA
    # --------------------------------------------------------

    s2_features = extract_spatial_features(
        images,
        S2_INDICES,
        s2_mean,
        s2_std,
        S2_WAVELENGTHS,
    )

    # --------------------------------------------------------
    # Verify current batch
    # --------------------------------------------------------

    expected_batch_shape = (
        len(batch_paths),
        196,
        768,
    )

    if tuple(s1_features.shape) != expected_batch_shape:
        raise RuntimeError(
            f"Unexpected S1 feature shape: "
            f"{tuple(s1_features.shape)}"
        )

    if tuple(s2_features.shape) != expected_batch_shape:
        raise RuntimeError(
            f"Unexpected S2 feature shape: "
            f"{tuple(s2_features.shape)}"
        )

    if not torch.isfinite(
        s1_features
    ).all().item():
        raise RuntimeError(
            "S1 feature tensor contains "
            "NaN or Inf."
        )

    if not torch.isfinite(
        s2_features
    ).all().item():
        raise RuntimeError(
            "S2 feature tensor contains "
            "NaN or Inf."
        )

    # --------------------------------------------------------
    # Move feature tensors back to CPU
    # --------------------------------------------------------

    shard_s1.append(
        s1_features.cpu()
    )

    shard_s2.append(
        s2_features.cpu()
    )

    shard_patch_ids.extend(
        patch_ids
    )

    processed += len(batch_paths)

    # --------------------------------------------------------
    # Write shard when it reaches SHARD_SIZE
    # --------------------------------------------------------

    if (
        len(shard_patch_ids) >= SHARD_SIZE
        or processed == total_samples
    ):

        s1_tensor = torch.cat(
            shard_s1,
            dim=0,
        )

        s2_tensor = torch.cat(
            shard_s2,
            dim=0,
        )

        # ----------------------------------------------------
        # Final shard sanity check
        # ----------------------------------------------------

        if s1_tensor.shape[0] != len(
            shard_patch_ids
        ):
            raise RuntimeError(
                "S1 sample count does not "
                "match patch ID count."
            )

        if s2_tensor.shape[0] != len(
            shard_patch_ids
        ):
            raise RuntimeError(
                "S2 sample count does not "
                "match patch ID count."
            )

        shard_path = os.path.join(
            OUTPUT_DIR,
            f"shard_{shard_id:04d}.pt"
        )

        torch.save(
            {
                "patch_ids": shard_patch_ids,
                "s1": s1_tensor,
                "s2": s2_tensor,
            },
            shard_path,
        )

        shard_size_mb = (
            os.path.getsize(shard_path)
            / 1024**2
        )

        print("\n" + "-" * 70)
        print(
            f"SHARD {shard_id:04d} SAVED"
        )
        print(
            "Samples:",
            len(shard_patch_ids)
        )
        print(
            "S1 shape:",
            tuple(s1_tensor.shape)
        )
        print(
            "S2 shape:",
            tuple(s2_tensor.shape)
        )
        print(
            "S1 dtype:",
            s1_tensor.dtype
        )
        print(
            "S2 dtype:",
            s2_tensor.dtype
        )
        print(
            "File size:",
            f"{shard_size_mb:.2f} MiB"
        )

        # ----------------------------------------------------
        # Reset shard buffers
        # ----------------------------------------------------

        shard_s1 = []
        shard_s2 = []
        shard_patch_ids = []

        shard_id += 1

    # --------------------------------------------------------
    # Progress information
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    rate = (
        processed / elapsed
        if elapsed > 0
        else 0
    )

    remaining = (
        total_samples - processed
    )

    eta_seconds = (
        remaining / rate
        if rate > 0
        else 0
    )

    eta_minutes = (
        eta_seconds / 60
    )

    print(
        f"Processed "
        f"{processed}/{total_samples} "
        f"({processed / total_samples * 100:.2f}%) "
        f"| "
        f"{rate:.2f} samples/s "
        f"| "
        f"ETA {eta_minutes:.1f} min"
    )

    # --------------------------------------------------------
    # Clear unused GPU memory
    # --------------------------------------------------------

    torch.cuda.empty_cache()


# ============================================================
# FINAL VERIFICATION
# ============================================================

elapsed = time.time() - start_time

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(
    "Total samples processed:",
    processed
)

print(
    "Total shards:",
    shard_id
)

print(
    "Elapsed time:",
    f"{elapsed / 60:.2f} minutes"
)

print(
    "Output directory:",
    OUTPUT_DIR
)


# ============================================================
# VERIFY SAVED SHARDS
# ============================================================

print("\nVerifying saved shards...")

saved_shards = sorted(
    glob.glob(
        os.path.join(
            OUTPUT_DIR,
            "shard_*.pt"
        )
    )
)

print(
    "Shard files found:",
    len(saved_shards)
)

total_saved = 0

for shard_path in saved_shards:

    shard = torch.load(
        shard_path,
        map_location="cpu",
    )

    patch_ids = shard["patch_ids"]
    s1 = shard["s1"]
    s2 = shard["s2"]

    expected_count = len(
        patch_ids
    )

    if tuple(s1.shape) != (
        expected_count,
        196,
        768,
    ):
        raise RuntimeError(
            f"Invalid S1 shape in "
            f"{shard_path}: "
            f"{tuple(s1.shape)}"
        )

    if tuple(s2.shape) != (
        expected_count,
        196,
        768,
    ):
        raise RuntimeError(
            f"Invalid S2 shape in "
            f"{shard_path}: "
            f"{tuple(s2.shape)}"
        )

    if s1.dtype != torch.float16:
        raise RuntimeError(
            f"S1 dtype is not float16 "
            f"in {shard_path}"
        )

    if s2.dtype != torch.float16:
        raise RuntimeError(
            f"S2 dtype is not float16 "
            f"in {shard_path}"
        )

    if not torch.isfinite(s1).all().item():
        raise RuntimeError(
            f"Non-finite S1 values in "
            f"{shard_path}"
        )

    if not torch.isfinite(s2).all().item():
        raise RuntimeError(
            f"Non-finite S2 values in "
            f"{shard_path}"
        )

    total_saved += expected_count

    print(
        os.path.basename(shard_path),
        "OK",
        "| samples:",
        expected_count
    )


print("\nTotal samples saved:", total_saved)

if total_saved != 10000:
    raise RuntimeError(
        f"Expected 10,000 saved samples, "
        f"found {total_saved}."
    )

print("\n" + "=" * 70)
print("ALL SHARDS VERIFIED SUCCESSFULLY")
print("=" * 70)