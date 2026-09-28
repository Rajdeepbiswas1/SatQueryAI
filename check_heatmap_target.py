import pandas as pd
import numpy as np


CSV_PATH = r"D:\SatQueryAI\data\processed\c1_diagnostic\exact_c1_per_sample.csv"


def build_target_heatmap(box, grid_size):
    x1, y1, x2, y2 = box

    edges = np.linspace(0.0, 1.0, grid_size + 1)

    total_box_area = (x2 - x1) * (y2 - y1)

    if total_box_area <= 0:
        return np.zeros((grid_size, grid_size), dtype=np.float64)

    heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)

    for j in range(grid_size):
        cell_y1 = edges[j]
        cell_y2 = edges[j + 1]

        intersection_h = max(
            0.0,
            min(cell_y2, y2) - max(cell_y1, y1)
        )

        if intersection_h <= 0:
            continue

        for i in range(grid_size):
            cell_x1 = edges[i]
            cell_x2 = edges[i + 1]

            intersection_w = max(
                0.0,
                min(cell_x2, x2) - max(cell_x1, x1)
            )

            if intersection_w <= 0:
                continue

            heatmap[j, i] = intersection_w * intersection_h

    heatmap_sum = heatmap.sum()

    if heatmap_sum > 0:
        heatmap /= heatmap_sum

    return heatmap


def main():
    df = pd.read_csv(CSV_PATH)

    boxes = df[
        [
            "target_x1",
            "target_y1",
            "target_x2",
            "target_y2",
        ]
    ].to_numpy(dtype=np.float64)

    print("=" * 70)
    print("TARGET HEATMAP DISCRETIZATION ANALYSIS")
    print("=" * 70)
    print(f"Rows: {len(boxes)}")
    print()

    for grid_size in [14, 28, 56]:

        nonzero_counts = []
        max_masses = []
        entropies = []

        for box in boxes:

            heatmap = build_target_heatmap(
                box,
                grid_size
            )

            nonzero = heatmap[heatmap > 0]

            nonzero_counts.append(
                len(nonzero)
            )

            max_masses.append(
                heatmap.max()
            )

            entropy = -np.sum(
                nonzero * np.log(
                    nonzero + 1e-12
                )
            )

            entropies.append(
                entropy
            )

        nonzero_counts = np.asarray(
            nonzero_counts
        )

        max_masses = np.asarray(
            max_masses
        )

        entropies = np.asarray(
            entropies
        )

        print(f"{grid_size}x{grid_size}:")
        print(
            f"  Mean nonzero cells     = "
            f"{nonzero_counts.mean():.3f}"
        )
        print(
            f"  Median nonzero cells   = "
            f"{np.median(nonzero_counts):.3f}"
        )
        print(
            f"  1-cell targets        = "
            f"{(nonzero_counts == 1).mean() * 100:.2f}%"
        )
        print(
            f"  Mean max-cell mass     = "
            f"{max_masses.mean():.6f}"
        )
        print(
            f"  Median max-cell mass   = "
            f"{np.median(max_masses):.6f}"
        )
        print(
            f"  Mean entropy           = "
            f"{entropies.mean():.6f}"
        )
        print(
            f"  Median entropy         = "
            f"{np.median(entropies):.6f}"
        )
        print()


if __name__ == "__main__":
    main()