from pathlib import Path

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(r"D:\SatQueryAI")

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed_test"
    / "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_83_16.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = torch.load(
    INPUT_FILE,
    map_location="cpu",
)

image = data["image"]
channel_names = data["channel_names"]

print("Tensor shape:", tuple(image.shape))
print("Channels:", channel_names)


# Channel indices
vh = image[0].numpy()
vv = image[1].numpy()

# S2 RGB = B04, B03, B02
# indices:
# B04 = 5
# B03 = 4
# B02 = 3
red = image[5].numpy()
green = image[4].numpy()
blue = image[3].numpy()

rgb = torch.tensor(
    [red, green, blue]
).permute(1, 2, 0).numpy()

rgb = rgb.clip(0, 1)


# ---------------------------------------------------------
# RGB
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title("Sentinel-2 RGB (B04/B03/B02)")
plt.axis("off")

rgb_file = OUTPUT_DIR / "test_rgb.png"
plt.savefig(
    rgb_file,
    bbox_inches="tight",
    dpi=150,
)
plt.close()

print("Saved:", rgb_file)


# ---------------------------------------------------------
# VH
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(vh)
plt.title("Sentinel-1 VH")
plt.axis("off")
plt.colorbar(label="Normalized value")

vh_file = OUTPUT_DIR / "test_vh.png"
plt.savefig(
    vh_file,
    bbox_inches="tight",
    dpi=150,
)
plt.close()

print("Saved:", vh_file)


# ---------------------------------------------------------
# VV
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(vv)
plt.title("Sentinel-1 VV")
plt.axis("off")
plt.colorbar(label="Normalized value")

vv_file = OUTPUT_DIR / "test_vv.png"
plt.savefig(
    vv_file,
    bbox_inches="tight",
    dpi=150,
)
plt.close()

print("Saved:", vv_file)


print("\nVisualization complete.")