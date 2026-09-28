from __future__ import annotations

import tarfile
from pathlib import Path


ROOT = Path(r"D:\SatQueryAI")

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
    / "c3_real_map_tests"
)

PATCH_IDS = [
    "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_42_82",
    "S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL_12_10",
    "S2A_MSIL2A_20171121T112351_N9999_R037_T29SND_73_85",
    "S2A_MSIL2A_20180506T100031_N9999_R122_T33UWP_33_87",
    "S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA_45_05",
    "S2B_MSIL2A_20170924T093019_N9999_R136_T35UMA_02_65",
    "S2B_MSIL2A_20171112T114339_N9999_R123_T29UPU_06_44",
    "S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_31_08",
    "S2B_MSIL2A_20180515T112109_N9999_R037_T29SNC_05_02",
    "S2B_MSIL2A_20180525T094029_N9999_R036_T35VNL_90_27",
]


def main() -> None:

    print("=" * 70)
    print("EXTRACT C3 REAL TEST MAPS")
    print("=" * 70)

    if not TAR_PATH.exists():
        raise FileNotFoundError(
            f"TAR not found:\n{TAR_PATH}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = {
        f"{patch_id}_reference_map.tif": patch_id
        for patch_id in PATCH_IDS
    }

    found = {}
    extracted = []

    print(f"TAR: {TAR_PATH}")
    print(f"Target maps: {len(PATCH_IDS)}")
    print("\nScanning TAR...")

    with tarfile.open(
        TAR_PATH,
        mode="r:",
    ) as tar:

        for member in tar:

            if not member.isfile():
                continue

            name = member.name.replace("\\", "/")
            basename = Path(name).name

            if basename not in targets:
                continue

            patch_id = targets[basename]

            if patch_id in found:
                continue

            print("\nFOUND:")
            print(name)

            # Expected official structure:
            #
            # Reference_Maps/
            #   <tile_id>/
            #     <patch_id>/
            #       <patch_id>_reference_map.tif

            if f"/{patch_id}/{basename}" not in f"/{name}":
                print("WARNING: unexpected path structure")

            destination = OUTPUT_ROOT / Path(name)

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fileobj = tar.extractfile(member)

            if fileobj is None:
                raise RuntimeError(
                    f"Could not read TAR member:\n{name}"
                )

            with destination.open(
                "wb"
            ) as out:

                while True:

                    chunk = fileobj.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    out.write(chunk)

            found[patch_id] = name
            extracted.append(destination)

            print(f"EXTRACTED: {destination}")

            if len(found) == len(PATCH_IDS):
                break

    print("\n" + "=" * 70)
    print("EXTRACTION RESULT")
    print("=" * 70)

    print(f"Requested: {len(PATCH_IDS)}")
    print(f"Found:     {len(found)}")
    print(f"Extracted: {len(extracted)}")
    print(f"Missing:   {len(PATCH_IDS) - len(found)}")

    if len(found) != len(PATCH_IDS):

        print("\nMISSING PATCHES:")

        for patch_id in PATCH_IDS:
            if patch_id not in found:
                print(patch_id)

        raise RuntimeError(
            "Not all requested maps were found."
        )

    print("\nAll requested maps extracted successfully.")


if __name__ == "__main__":
    main()