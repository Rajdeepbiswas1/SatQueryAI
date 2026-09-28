import os
import rasterio

folder = (
    r"D:\SatQueryAI\data\processed\images_eval\BigEarthNet-S2"
    r"\S2B_MSIL2A_20170906T101019_N9999_R022_T34WFS"
    r"\S2B_MSIL2A_20170906T101019_N9999_R022_T34WFS_39_32"
)

bands = [
    "B04", "B03", "B02",
    "B05", "B06", "B07",
    "B08", "B11", "B12"
]

print("Band       Min       Max       Mean")
print("-" * 40)

for band in bands:
    filename = None

    for name in os.listdir(folder):
        if name.endswith(f"_{band}.tif"):
            filename = os.path.join(folder, name)
            break

    if filename is None:
        print(f"{band:>4}   NOT FOUND")
        continue

    with rasterio.open(filename) as src:
        image = src.read(1)

    print(
        f"{band:>4} "
        f"{image.min():9.2f} "
        f"{image.max():9.2f} "
        f"{image.mean():10.2f}"
    )