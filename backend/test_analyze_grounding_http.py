from pathlib import Path
import subprocess
import sys

from preprocessing.preprocess_dataset import (
    build_s1_paths,
    build_s2_paths,
)


BASE_URL = "http://127.0.0.1:8000/analyze"

PATCH_ID = (
    "S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU_83_16"
)

S1_NAME = (
    "S1A_IW_GRDH_1SDV_20180526T045642_34WFU_83_16"
)

QUERY = "Where is the largest urban area in the image?"


def main() -> None:

    s1_paths = build_s1_paths(S1_NAME)
    s2_paths = build_s2_paths(PATCH_ID)

    paths = (
        list(s1_paths.values())
        + list(s2_paths.values())
    )

    print("SATQUERYAI — PRODUCTION GROUNDING HTTP TEST")
    print("=" * 80)
    print()
    print(f"Endpoint: {BASE_URL}")
    print(f"Patch ID: {PATCH_ID}")
    print(f"Query: {QUERY}")
    print(f"Files: {len(paths)}")
    print()

    # ---------------------------------------------------------
    # Verify every local path before calling curl.
    # ---------------------------------------------------------

    missing = [
        str(path)
        for path in paths
        if not path.is_file()
    ]

    if missing:
        print("MISSING FILES:")
        for path in missing:
            print(" ", path)

        raise RuntimeError(
            f"{len(missing)} required files are missing."
        )

    print("All 14 files found: PASS")
    print()

    # ---------------------------------------------------------
    # Build curl arguments directly.
    # No shell interpolation.
    # No manually typed long paths.
    # ---------------------------------------------------------

    command = [
        "curl.exe",
        "-sS",
        "-X",
        "POST",
        BASE_URL,
        "-F",
        f"query={QUERY}",
        "-F",
        f"patch_id={PATCH_ID}",
    ]

    for path in paths:
        command.extend(
            [
                "-F",
                f"image_files=@{path}",
            ]
        )

    print("Sending request...")
    print()

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if completed.returncode != 0:

        print(
            "curl failed with exit code:",
            completed.returncode,
        )

        if completed.stderr:
            print(
                completed.stderr
            )

        raise RuntimeError(
            "Production HTTP request failed."
        )

    print(completed.stdout)

    if completed.stderr:
        print(
            "\nCURL STDERR:\n",
            completed.stderr,
        )

    print()
    print("=" * 80)
    print("GROUNDING HTTP REQUEST COMPLETED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(
            "GROUNDING HTTP TEST: FAIL"
        )
        print(
            str(exc)
        )
        sys.exit(1)