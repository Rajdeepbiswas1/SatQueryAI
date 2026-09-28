from pathlib import Path
import argparse
import subprocess
import tarfile
import sys
import time


SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")


def safe_output_path(base_dir: Path, member_name: str) -> Path:
    relative = Path(member_name)

    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"Unsafe archive path: {member_name}"
        )

    output = base_dir / relative

    try:
        output.resolve().relative_to(
            base_dir.resolve()
        )
    except ValueError:
        raise RuntimeError(
            f"Unsafe output path: {member_name}"
        )

    return output


def extract(
    archive: Path,
    file_list: Path,
    output_dir: Path,
) -> None:

    if not archive.exists():
        raise FileNotFoundError(archive)

    if not file_list.exists():
        raise FileNotFoundError(file_list)

    if not SEVEN_ZIP.exists():
        raise FileNotFoundError(SEVEN_ZIP)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested = {
        line.strip()
        for line in file_list.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    if not requested:
        raise RuntimeError(
            "The requested file list is empty."
        )

    print(
        f"Requested files: {len(requested):,}"
    )
    print(f"Archive: {archive}")
    print(f"Output:  {output_dir}")
    print()

    found = set()
    members_scanned = 0
    files_extracted = 0

    start = time.time()
    last_report = start

    process = subprocess.Popen(
        [
            str(SEVEN_ZIP),
            "x",
            str(archive),
            "-so",
        ],
        stdout=subprocess.PIPE,
        stderr=None,
    )

    assert process.stdout is not None

    try:

        with tarfile.open(
            fileobj=process.stdout,
            mode="r|",
        ) as tar:

            for member in tar:

                members_scanned += 1

                if member.name not in requested:
                    continue

                if not member.isfile():
                    continue

                found.add(member.name)

                output_path = safe_output_path(
                    output_dir,
                    member.name,
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if (
                    output_path.exists()
                    and output_path.stat().st_size
                    == member.size
                ):
                    continue

                source = tar.extractfile(member)

                if source is None:
                    print(
                        f"WARNING: Could not read "
                        f"{member.name}"
                    )
                    continue

                with source, open(
                    output_path,
                    "wb",
                ) as destination:

                    while True:

                        chunk = source.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        destination.write(chunk)

                files_extracted += 1

                elapsed = time.time() - start

                if elapsed - last_report >= 5:

                    print(
                        f"Scanned: {members_scanned:,} | "
                        f"Matched: "
                        f"{len(found):,}/"
                        f"{len(requested):,} | "
                        f"Newly written: "
                        f"{files_extracted:,}",
                        flush=True,
                    )

                    last_report = elapsed

    finally:

        process.stdout.close()

    return_code = process.wait()

    if return_code != 0:

        raise RuntimeError(
            f"7-Zip exited with return code "
            f"{return_code}."
        )

    missing = requested - found

    elapsed = time.time() - start

    print()
    print("=== Finished ===")
    print(
        f"Archive members scanned: "
        f"{members_scanned:,}"
    )
    print(
        f"Requested:               "
        f"{len(requested):,}"
    )
    print(
        f"Found:                   "
        f"{len(found):,}"
    )
    print(
        f"Newly written:           "
        f"{files_extracted:,}"
    )
    print(
        f"Missing:                 "
        f"{len(missing):,}"
    )
    print(
        f"Time:                    "
        f"{elapsed / 60:.1f} minutes"
    )

    if missing:

        print("\nMissing files:")

        for name in sorted(missing):
            print(name)

        sys.exit(1)

    print("\nSuccess.")


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--archive",
        required=True,
    )

    parser.add_argument(
        "--list",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    extract(
        Path(args.archive),
        Path(args.list),
        Path(args.output),
    )


if __name__ == "__main__":
    main()