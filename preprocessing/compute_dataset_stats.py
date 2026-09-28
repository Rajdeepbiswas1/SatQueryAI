from pathlib import Path
import re

import numpy as np
import rasterio


PROJECT_ROOT = Path(r"D:\SatQueryAI")

S1_LIST = PROJECT_ROOT / "data" / "raw" / "s1_extract_list.txt"
S2_LIST = PROJECT_ROOT / "data" / "raw" / "s2_extract_list.txt"

IMAGE_ROOT = PROJECT_ROOT / "data" / "processed" / "images_raw"

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_stats.txt"

# Number of values retained per channel for percentile estimation.
# Mean/std/min/max are calculated over all pixels.
RESERVOIR_SIZE = 200_000

RNG = np.random.default_rng(42)


def get_channel(path: str) -> str:
    name = Path(path).name

    match = re.search(
        r"_((?:VH|VV|B01|B02|B03|B04|B05|B06|B07|B08|B09|B11|B12|B8A))\.tif$",
        name,
    )

    if not match:
        raise ValueError(f"Could not determine channel: {path}")

    return match.group(1)


def update_reservoir(
    reservoir: np.ndarray,
    seen: int,
    values: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Reservoir-sample values so memory stays bounded.
    """
    values = np.asarray(values, dtype=np.float64).ravel()

    for value in values:
        seen += 1

        if seen <= len(reservoir):
            reservoir[seen - 1] = value
        else:
            j = RNG.integers(0, seen)

            if j < len(reservoir):
                reservoir[j] = value

    return reservoir, seen


def initialize_stats():
    channels = [
        "VH", "VV",
        "B01", "B02", "B03", "B04",
        "B05", "B06", "B07", "B08",
        "B09", "B11", "B12", "B8A",
    ]

    stats = {}

    for channel in channels:
        stats[channel] = {
            "count": 0,
            "sum": 0.0,
            "sumsq": 0.0,
            "min": np.inf,
            "max": -np.inf,
            "zeros": 0,
            "seen_reservoir": 0,
            "reservoir": np.empty(RESERVOIR_SIZE, dtype=np.float64),
        }

    return stats


def process_file(path: Path, channel: str, stats: dict) -> None:
    with rasterio.open(path) as src:
        data = src.read(1)

    values = data.astype(np.float64, copy=False).ravel()

    s = stats[channel]

    s["count"] += values.size
    s["sum"] += values.sum(dtype=np.float64)
    s["sumsq"] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
    s["min"] = min(s["min"], float(values.min()))
    s["max"] = max(s["max"], float(values.max()))
    s["zeros"] += int(np.count_nonzero(values == 0))

    reservoir, seen = update_reservoir(
        s["reservoir"],
        s["seen_reservoir"],
        values,
    )

    s["reservoir"] = reservoir
    s["seen_reservoir"] = seen


def finalize_channel(channel: str, s: dict) -> dict:
    count = s["count"]

    mean = s["sum"] / count

    variance = (s["sumsq"] / count) - (mean * mean)
    variance = max(variance, 0.0)

    std = np.sqrt(variance)

    reservoir_count = min(s["seen_reservoir"], RESERVOIR_SIZE)
    sample = np.sort(s["reservoir"][:reservoir_count])

    percentiles = np.percentile(
        sample,
        [1, 5, 50, 95, 99],
    )

    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": s["min"],
        "max": s["max"],
        "zero_pct": 100.0 * s["zeros"] / count,
        "p01": percentiles[0],
        "p05": percentiles[1],
        "p50": percentiles[2],
        "p95": percentiles[3],
        "p99": percentiles[4],
    }


def process_list(list_file: Path, stats: dict) -> None:
    lines = [
        line.strip()
        for line in list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    total = len(lines)

    print(f"\nProcessing {list_file.name}")
    print(f"Files: {total:,}")

    for index, relative_path in enumerate(lines, start=1):

        channel = get_channel(relative_path)
        path = IMAGE_ROOT / Path(relative_path)

        if not path.is_file():
            raise FileNotFoundError(path)

        process_file(path, channel, stats)

        if index % 1000 == 0 or index == total:
            print(
                f"  {index:,}/{total:,} files processed",
                flush=True,
            )


def main():
    stats = initialize_stats()

    print("=== SatQueryAI dataset statistics ===")
    print("This will read all selected S1 and S2 TIFF files.")
    print()

    process_list(S1_LIST, stats)
    process_list(S2_LIST, stats)

    results = {}

    for channel, channel_stats in stats.items():
        results[channel] = finalize_channel(
            channel,
            channel_stats,
        )

    print("\n=== Results ===\n")

    lines = []

    header = (
        "Channel | Pixels | Mean | Std | Min | Max | "
        "P01 | P05 | P50 | P95 | P99 | Zero%"
    )

    print(header)
    print("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for channel in [
        "VH", "VV",
        "B01", "B02", "B03", "B04",
        "B05", "B06", "B07", "B08",
        "B09", "B11", "B12", "B8A",
    ]:
        r = results[channel]

        line = (
            f"{channel:>4} | "
            f"{r['count']:>12,} | "
            f"{r['mean']:>10.4f} | "
            f"{r['std']:>10.4f} | "
            f"{r['min']:>10.4f} | "
            f"{r['max']:>10.4f} | "
            f"{r['p01']:>10.4f} | "
            f"{r['p05']:>10.4f} | "
            f"{r['p50']:>10.4f} | "
            f"{r['p95']:>10.4f} | "
            f"{r['p99']:>10.4f} | "
            f"{r['zero_pct']:>8.4f}"
        )

        print(line)
        lines.append(line)

    OUTPUT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved statistics to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
    