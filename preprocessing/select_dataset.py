from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PARQUET_FILE = Path(
    r"D:\SatQueryAI\data\raw\BigEarthNet.txt.parquet"
)

OUTPUT_FILE = Path(
    r"D:\SatQueryAI\data\splits\selected_pairs.csv"
)

TARGET_PAIRS = 10_000
RANDOM_SEED = 42


# ---------------------------------------------------------
# Load only the columns needed for selecting image pairs
# ---------------------------------------------------------

print("Reading dataset metadata...")

table = pq.read_table(
    PARQUET_FILE,
    columns=[
        "patch_id",
        "split",
        "country",
        "season",
        "climate_zone",
    ],
)

df = table.to_pandas()

# We only want the official training split.
df = df[df["split"] == "train"]

# One row per unique image pair.
df = df.drop_duplicates(subset="patch_id").reset_index(drop=True)

print(f"Available training pairs: {len(df):,}")


# ---------------------------------------------------------
# Validate
# ---------------------------------------------------------

if TARGET_PAIRS > len(df):
    raise ValueError(
        f"Requested {TARGET_PAIRS:,} pairs, "
        f"but only {len(df):,} training pairs are available."
    )


# ---------------------------------------------------------
# Country/season-aware proportional sampling
# ---------------------------------------------------------
#
# We sample within each country, preserving that country's
# natural seasonal distribution as much as possible.
#
# Countries with more available data receive more samples,
# while every country is represented when possible.
# ---------------------------------------------------------

country_counts = df["country"].value_counts()

# Initial proportional allocation.
allocations = (
    country_counts / country_counts.sum() * TARGET_PAIRS
).round().astype(int)

# Make sure the allocation sums exactly to TARGET_PAIRS.
difference = TARGET_PAIRS - allocations.sum()

if difference != 0:
    # Add/subtract the difference from the largest countries.
    sorted_countries = country_counts.sort_values(
        ascending=False
    ).index.tolist()

    step = 1 if difference > 0 else -1

    for i in range(abs(difference)):
        country = sorted_countries[i % len(sorted_countries)]
        allocations[country] += step


# ---------------------------------------------------------
# Sample within each country while preserving season
# ---------------------------------------------------------

selected_parts = []

for country, country_target in allocations.items():

    country_df = df[df["country"] == country].copy()

    if country_target <= 0:
        continue

    # Desired number from each season based on the country's
    # actual seasonal distribution.
    season_counts = country_df["season"].value_counts()

    season_allocations = (
        season_counts / season_counts.sum() * country_target
    ).round().astype(int)

    # Correct rounding so the total is exactly country_target.
    season_difference = country_target - season_allocations.sum()

    if season_difference != 0:
        seasons = season_counts.sort_values(
            ascending=False
        ).index.tolist()

        step = 1 if season_difference > 0 else -1

        for i in range(abs(season_difference)):
            season = seasons[i % len(seasons)]
            season_allocations[season] += step

    for season, season_target in season_allocations.items():

        if season_target <= 0:
            continue

        season_df = country_df[
            country_df["season"] == season
        ]

        # Never ask for more samples than actually exist.
        season_target = min(
            season_target,
            len(season_df)
        )

        sampled = season_df.sample(
            n=season_target,
            random_state=RANDOM_SEED,
        )

        selected_parts.append(sampled)


# ---------------------------------------------------------
# Combine all selected pairs
# ---------------------------------------------------------

selected = (
    pd.concat(selected_parts)
    .drop_duplicates(subset="patch_id")
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Safety correction
# ---------------------------------------------------------

# Rounding can occasionally produce one or two too many/few.
if len(selected) > TARGET_PAIRS:

    selected = selected.sample(
        n=TARGET_PAIRS,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

elif len(selected) < TARGET_PAIRS:

    remaining = df[
        ~df["patch_id"].isin(selected["patch_id"])
    ]

    extra = remaining.sample(
        n=TARGET_PAIRS - len(selected),
        random_state=RANDOM_SEED,
    )

    selected = pd.concat(
        [selected, extra],
        ignore_index=True,
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

selected.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("Selection complete.")
print(f"Selected pairs: {len(selected):,}")
print(f"Saved to: {OUTPUT_FILE}")

print()
print("Country distribution:")
print(selected["country"].value_counts())

print()
print("Season distribution:")
print(selected["season"].value_counts())

print()
print("Climate distribution:")
print(selected["climate_zone"].value_counts())