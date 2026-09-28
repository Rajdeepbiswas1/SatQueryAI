import os
import glob
import torch

S2_INDICES = [5, 4, 3, 6, 7, 8, 9, 11, 12]
S2_BANDS = ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]

DATA_DIR = r"D:\SatQueryAI\data\processed\preprocessed"
OUTPUT = r"D:\SatQueryAI\data\processed\dofa_s2_stats.pt"


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))

    if not paths:
        raise RuntimeError("No .pt files found.")

    print("=" * 70)
    print("COMPUTING DOFA S2 ADAPTER STATISTICS")
    print("=" * 70)
    print("Samples:", len(paths))

    sum_x = torch.zeros(9, dtype=torch.float64)
    sum_x2 = torch.zeros(9, dtype=torch.float64)
    count = 0

    for i, path in enumerate(paths, 1):
        data = torch.load(path, map_location="cpu")
        image = data["image"].float()

        s2 = image[S2_INDICES]  # [9,120,120]

        flat = s2.reshape(9, -1).double()

        sum_x += flat.sum(dim=1)
        sum_x2 += (flat * flat).sum(dim=1)
        count += flat.shape[1]

        if i % 500 == 0 or i == len(paths):
            print(f"Processed {i}/{len(paths)}")

    mean = sum_x / count
    variance = (sum_x2 / count) - mean.square()
    std = torch.sqrt(torch.clamp(variance, min=1e-12))

    stats = {
        "bands": S2_BANDS,
        "indices": S2_INDICES,
        "mean": mean.float(),
        "std": std.float(),
        "num_images": len(paths),
    }

    torch.save(stats, OUTPUT)

    print("\nResults:")
    for band, m, s in zip(S2_BANDS, mean, std):
        print(f"{band}: mean={m:.6f}, std={s:.6f}")

    print("\nSaved:")
    print(OUTPUT)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()