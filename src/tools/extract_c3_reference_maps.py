from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"D:\SatQueryAI")

TRAIN_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_train.parquet"
)

VAL_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "reference_grounding_split"
    / "reference_val.parquet"
)

TAR_PATH = (
    ROOT
    / "data"
    / "raw"
    / "BigEarthNet_v2"
    / "temp_reference_maps"
    / "Reference_Maps.tar"
)

OUTPUT_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "c3_reference_maps"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_PATH = (
    OUTPUT_ROOT
    / "extraction_manifest.json"
)


# ============================================================
# LOAD PATCH IDS
# ============================================================

def load_patch_ids(
    parquet_path: Path,
    split_name: str,
) -> tuple[set[str], dict[str, str]]:

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{split_name} parquet not found:\n"
            f"{parquet_path}"
        )

    df = pd.read_parquet(
        parquet_path
    )

    if "patch_id" not in df.columns:
        raise ValueError(
            f"'patch_id' column missing from "
            f"{split_name} parquet.\n"
            f"Columns: {list(df.columns)}"
        )

    patch_ids = (
        df["patch_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )

    split_map = {
        patch_id: split_name
        for patch_id in patch_ids
    }

    return set(patch_ids), split_map


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("C3 BULK REFERENCE-MAP EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check source TAR
    # --------------------------------------------------------

    if not TAR_PATH.exists():
        raise FileNotFoundError(
            f"Reference-map TAR not found:\n"
            f"{TAR_PATH}"
        )

    # --------------------------------------------------------
    # Load train/validation patch IDs
    # --------------------------------------------------------

    print("\nLoading reference split patch IDs...")

    train_ids, train_map = load_patch_ids(
        TRAIN_PARQUET,
        "train",
    )

    val_ids, val_map = load_patch_ids(
        VAL_PARQUET,
        "val",
    )

    # --------------------------------------------------------
    # Ensure train/val do not overlap
    # --------------------------------------------------------

    overlap = (
        train_ids
        & val_ids
    )

    print(
        f"Train unique patches: {len(train_ids)}"
    )

    print(
        f"Validation unique patches: {len(val_ids)}"
    )

    print(
        f"Train/validation overlap: {len(overlap)}"
    )

    if overlap:
        sample = sorted(overlap)[:10]

        raise RuntimeError(
            "Train/validation patch overlap detected.\n"
            f"Sample: {sample}"
        )

    # --------------------------------------------------------
    # Combined expected patches
    # --------------------------------------------------------

    patch_to_split = {}

    patch_to_split.update(
        train_map
    )

    patch_to_split.update(
        val_map
    )

    expected_patch_ids = set(
        patch_to_split.keys()
    )

    expected_basenames = {
        f"{patch_id}_reference_map.tif": patch_id
        for patch_id in expected_patch_ids
    }

    print(
        f"Total unique patches requested: "
        f"{len(expected_patch_ids)}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    print(
        f"Output directory:\n"
        f"{OUTPUT_ROOT}"
    )

    # --------------------------------------------------------
    # Scan TAR and extract exact members
    # --------------------------------------------------------

    found_patch_ids = set()
    extracted_paths = []
    duplicate_members = []

    print("\nOpening TAR...")
    print(
        "The TAR will be scanned sequentially. "
        "Original archive will not be modified."
    )

    with tarfile.open(
        TAR_PATH,
        mode="r:",
    ) as tar:

        for member in tar:

            if not member.isfile():
                continue

            name = member.name.replace(
                "\\",
                "/",
            )

            basename = Path(name).name

            if basename not in expected_basenames:
                continue

            patch_id = expected_basenames[
                basename
            ]

            if patch_id in found_patch_ids:

                duplicate_members.append(
                    name
                )

                continue

            # ------------------------------------------------
            # Verify expected naming relationship
            # ------------------------------------------------

            expected_filename = (
                f"{patch_id}_reference_map.tif"
            )

            if basename != expected_filename:
                raise RuntimeError(
                    "Unexpected filename relationship:\n"
                    f"Patch ID: {patch_id}\n"
                    f"Member: {name}"
                )

            # ------------------------------------------------
            # Preserve official archive structure
            # ------------------------------------------------

            relative_path = Path(*Path(name).parts)

            destination = (
                OUTPUT_ROOT
                / relative_path
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ------------------------------------------------
            # Extract exact TAR member
            # ------------------------------------------------

            fileobj = tar.extractfile(
                member
            )

            if fileobj is None:
                raise RuntimeError(
                    f"Could not read TAR member:\n"
                    f"{name}"
                )

            with destination.open(
                "wb"
            ) as output_file:

                while True:

                    chunk = fileobj.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    output_file.write(
                        chunk
                    )

            found_patch_ids.add(
                patch_id
            )

            extracted_paths.append(
                {
                    "patch_id": patch_id,
                    "split": patch_to_split[
                        patch_id
                    ],
                    "archive_path": name,
                    "extracted_path": str(
                        destination
                    ),
                    "size_bytes": destination.stat().st_size,
                }
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            found_count = len(
                found_patch_ids
            )

            if (
                found_count <= 10
                or found_count % 500 == 0
                or found_count == len(
                    expected_patch_ids
                )
            ):

                print(
                    f"Extracted "
                    f"{found_count}/"
                    f"{len(expected_patch_ids)}"
                )

            # ------------------------------------------------
            # Stop early if every required map was found
            # ------------------------------------------------

            if (
                len(found_patch_ids)
                == len(expected_patch_ids)
            ):
                print(
                    "\nAll requested maps found."
                )
                break

    # ========================================================
    # MISSING PATCHES
    # ========================================================

    missing_patch_ids = sorted(
        expected_patch_ids
        - found_patch_ids
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("EXTRACTION RESULT")
    print("=" * 70)

    print(
        f"Requested patches: "
        f"{len(expected_patch_ids)}"
    )

    print(
        f"Found in TAR: "
        f"{len(found_patch_ids)}"
    )

    print(
        f"Extracted: "
        f"{len(extracted_paths)}"
    )

    print(
        f"Missing: "
        f"{len(missing_patch_ids)}"
    )

    print(
        f"Duplicate TAR members: "
        f"{len(duplicate_members)}"
    )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest = {
        "tar_path": str(TAR_PATH),
        "train_parquet": str(TRAIN_PARQUET),
        "val_parquet": str(VAL_PARQUET),
        "output_root": str(OUTPUT_ROOT),
        "train_unique_patches": len(train_ids),
        "val_unique_patches": len(val_ids),
        "train_val_overlap": len(overlap),
        "requested_unique_patches": len(
            expected_patch_ids
        ),
        "found_patches": len(
            found_patch_ids
        ),
        "extracted_maps": len(
            extracted_paths
        ),
        "missing_patches": len(
            missing_patch_ids
        ),
        "duplicate_tar_members": len(
            duplicate_members
        ),
        "missing_patch_ids": missing_patch_ids,
        "duplicate_members": duplicate_members,
        "maps": extracted_paths,
    }

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        f"\nManifest:\n"
        f"{RESULT_PATH}"
    )

    # --------------------------------------------------------
    # Hard failure if any expected maps are missing
    # --------------------------------------------------------

    if missing_patch_ids:

        print(
            "\nMISSING PATCH IDS:"
        )

        for patch_id in missing_patch_ids[:50]:
            print(
                patch_id
            )

        if len(missing_patch_ids) > 50:
            print(
                f"... and "
                f"{len(missing_patch_ids) - 50} more"
            )

        raise RuntimeError(
            "C3 reference-map extraction incomplete."
        )

    print("\n" + "=" * 70)
    print(
        "C3 REFERENCE-MAP EXTRACTION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()