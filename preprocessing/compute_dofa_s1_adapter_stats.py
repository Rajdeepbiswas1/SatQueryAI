import glob
import os
import torch


S1_INDICES = [0, 1]
S1_BANDS = ["VH", "VV"]

DATA_DIR = r"D:\SatQueryAI\data\processed\preprocessed"
OUTPUT = r"D:\SatQueryAI\data\processed\dofa_s1_stats.pt"


def main():
    paths = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.pt"))
    )

    if not paths:
        raise RuntimeError("No .pt files found.")

    print("=" * 70)
    print("COMPUTING DOFA S1 ADAPTER STATISTICS")
    print("=" * 70)
    print("Samples:", len(paths))

    sum_x = torch.zeros(
        len(S1_INDICES),
        dtype=torch.float64,
    )

    sum_x2 = torch.zeros(
        len(S1_INDICES),
        dtype=torch.float64,
    )

    count = 0

    for i, path in enumerate(paths, 1):
        data = torch.load(
            path,
            map_location="cpu",
        )

        image = data["image"].float()

        # Select VH and VV from the existing 14-channel tensor.
        s1 = image[S1_INDICES]

        # Flatten each channel independently.
        flat = s1.reshape(
            len(S1_INDICES),
            -1,
        ).double()

        sum_x += flat.sum(dim=1)
        sum_x2 += (flat * flat).sum(dim=1)

        count += flat.shape[1]

        if i % 500 == 0 or i == len(paths):
            print(
                f"Processed {i}/{len(paths)}"
            )

    mean = sum_x / count

    variance = (
        (sum_x2 / count)
        - mean.square()
    )

    std = torch.sqrt(
        torch.clamp(
            variance,
            min=1e-12,
        )
    )

    stats = {
        "bands": S1_BANDS,
        "indices": S1_INDICES,
        "mean": mean.float(),
        "std": std.float(),
        "num_images": len(paths),
    }

    torch.save(
        stats,
        OUTPUT,
    )

    print("\nResults:")

    for band, m, s in zip(
        S1_BANDS,
        mean,
        std,
    ):
        print(
            f"{band}: "
            f"mean={m:.6f}, "
            f"std={s:.6f}"
        )

    print("\nSaved:")
    print(OUTPUT)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()