import os
import pandas as pd


PROJECT_ROOT = r"D:\SatQueryAI"

MANIFEST = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "temporal_manifest.csv",
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
)

OUT_LIST = os.path.join(
    OUT_DIR,
    "second_temporal_extract_list.txt",
)


def main():
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    df = pd.read_csv(MANIFEST)

    if "pair_id" not in df.columns:
        raise ValueError(
            "Manifest does not contain pair_id column."
        )

    pair_ids = (
        df["pair_id"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print(f"Unique temporal pairs: {len(pair_ids)}")

    entries = []

    for pair_id in pair_ids:
        entries.extend([
            f"im1\\{pair_id}.png",
            f"im2\\{pair_id}.png",
            f"label1\\{pair_id}.png",
            f"label2\\{pair_id}.png",
        ])

    with open(
        OUT_LIST,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write("\n".join(entries))
        f.write("\n")

    print(f"Files requested: {len(entries)}")
    print(f"Expected: {len(pair_ids) * 4}")
    print(f"Extraction list: {OUT_LIST}")

    if len(entries) != len(pair_ids) * 4:
        raise RuntimeError(
            "Extraction-list size is incorrect."
        )


if __name__ == "__main__":
    main()