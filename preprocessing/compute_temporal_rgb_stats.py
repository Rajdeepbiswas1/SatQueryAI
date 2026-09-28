import glob
import os

import torch


DATA_DIR = r"D:\SatQueryAI\data\processed\temporal_preprocessed"
OUTPUT = r"D:\SatQueryAI\data\processed\temporal_rgb_stats.pt"


def compute_stats(paths, key):
    sum_x = torch.zeros(3, dtype=torch.float64)
    sum_x2 = torch.zeros(3, dtype=torch.float64)
    count = 0

    for i, path in enumerate(paths, 1):
        data = torch.load(path, map_location="cpu")
        image = data[key].float()  # [3, 512, 512]

        flat = image.reshape(3, -1).double()

        sum_x += flat.sum(dim=1)
        sum_x2 += (flat * flat).sum(dim=1)
        count += flat.shape[1]

        if i % 500 == 0 or i == len(paths):
            print(f"{key}: processed {i}/{len(paths)}")

    mean = sum_x / count

    variance = (sum_x2 / count) - mean.square()

    std = torch.sqrt(
        torch.clamp(
            variance,
            min=1e-12,
        )
    )

    return mean.float(), std.float()


def main():
    paths = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR,
                "*.pt",
            )
        )
    )

    if not paths:
        raise RuntimeError("No temporal .pt files found.")

    print("=" * 70)
    print("TEMPORAL RGB STATISTICS")
    print("=" * 70)
    print("Samples:", len(paths))

    t1_mean, t1_std = compute_stats(paths, "t1")
    t2_mean, t2_std = compute_stats(paths, "t2")

    stats = {
        "t1_mean": t1_mean,
        "t1_std": t1_std,
        "t2_mean": t2_mean,
        "t2_std": t2_std,
        "num_pairs": len(paths),
    }

    torch.save(stats, OUTPUT)

    print("\nT1:")
    print("mean:", t1_mean)
    print("std :", t1_std)

    print("\nT2:")
    print("mean:", t2_mean)
    print("std :", t2_std)

    print("\nSaved:")
    print(OUTPUT)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()