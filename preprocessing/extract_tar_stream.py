from pathlib import Path
import subprocess
import tarfile
import sys
import time


PROJECT_ROOT = Path(r"D:\SatQueryAI")

ARCHIVE = PROJECT_ROOT / "data" / "raw" / "BigEarthNet_v2" / "BigEarthNet-S2.tar.zst"
FILE_LIST = PROJECT_ROOT / "data" / "raw" / "s2_extract_list.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "images_raw"

SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")


def safe_output_path(base_dir: Path, member_name: str) -> Path:
    """
    Convert an archive member path into a safe local path.
    Prevents accidental path traversal.
    """
    relative = Path(member_name)

    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe archive path: {member_name}")

    output = base_dir / relative

    try:
        output.resolve().relative_to(base_dir.resolve())
    except ValueError:
        raise RuntimeError(f"Unsafe output path: {member_name}")

    return output


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)

    if not FILE_LIST.exists():
        raise FileNotFoundError(FILE_LIST)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    requested = {
        line.strip()
        for line in FILE_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    if not requested:
        raise RuntimeError("The requested file list is empty.")

    print(f"Requested files: {len(requested):,}")
    print(f"Archive: {ARCHIVE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    found = set()
    members_scanned = 0
    files_extracted = 0

    start = time.time()
    last_report = start

    # 7-Zip removes the outer Zstandard compression layer and
    # sends the resulting TAR bytes to stdout.
    process = subprocess.Popen(
        [
            str(SEVEN_ZIP),
            "x",
            str(ARCHIVE),
            "-so",
        ],
        stdout=subprocess.PIPE,
        stderr=None,
    )

    assert process.stdout is not None

    try:
        # r| means streaming TAR mode.
        # It does not require random access to the TAR.
        with tarfile.open(fileobj=process.stdout, mode="r|") as tar:

            for member in tar:
                members_scanned += 1

                if member.name not in requested:
                    continue

                if not member.isfile():
                    continue

                found.add(member.name)

                output_path = safe_output_path(OUTPUT_DIR, member.name)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Don't rewrite files that already exist with the same size.
                if output_path.exists() and output_path.stat().st_size == member.size:
                    continue

                source = tar.extractfile(member)

                if source is None:
                    print(f"WARNING: Could not read {member.name}")
                    continue

                with source, open(output_path, "wb") as destination:
                    while True:
                        chunk = source.read(1024 * 1024)

                        if not chunk:
                            break

                        destination.write(chunk)

                files_extracted += 1

                elapsed = time.time() - start

                if elapsed - last_report >= 5:
                    print(
                        f"Scanned: {members_scanned:,} | "
                        f"Matched: {len(found):,}/{len(requested):,} | "
                        f"Newly written: {files_extracted:,}",
                        flush=True,
                    )
                    last_report = elapsed


    finally:
        # Close the pipe so 7-Zip doesn't remain attached.
        process.stdout.close()

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"7-Zip exited with return code {return_code}."
        )

    missing = requested - found

    elapsed = time.time() - start

    print()
    print("=== Finished ===")
    print(f"Archive members scanned: {members_scanned:,}")
    print(f"Requested:               {len(requested):,}")
    print(f"Found:                   {len(found):,}")
    print(f"Newly written:           {files_extracted:,}")
    print(f"Missing:                 {len(missing):,}")
    print(f"Time:                    {elapsed / 60:.1f} minutes")

    if missing:
        print("\nMissing files:")
        for name in sorted(missing):
            print(name)

        sys.exit(1)

    print("\nSuccess.")


if __name__ == "__main__":
    main()