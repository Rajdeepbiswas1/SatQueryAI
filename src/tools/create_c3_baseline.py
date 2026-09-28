from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(r"D:\SatQueryAI")

RAW_ROOT = ROOT / "data" / "raw" / "BigEarthNet_v2"
PROCESSED_ROOT = ROOT / "data" / "processed"
CHECKPOINT_ROOT = ROOT / "models" / "checkpoints" / "c2_heatmap_only"

OUTPUT_ROOT = PROCESSED_ROOT / "c3_baseline"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_ROOT / "baseline_manifest.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def describe_file(path: Path, hash_file: bool = True) -> dict:
    item = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return item

    stat = path.stat()
    item["size_bytes"] = stat.st_size
    item["modified_time"] = datetime.fromtimestamp(
        stat.st_mtime, tz=timezone.utc
    ).isoformat()

    if hash_file and path.is_file():
        print(f"SHA256: {path}")
        item["sha256"] = sha256_file(path)

    return item


def describe_directory(path: Path) -> dict:
    item = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return item

    files = []
    total_bytes = 0

    for p in path.rglob("*"):
        if p.is_file():
            try:
                size = p.stat().st_size
                total_bytes += size
                files.append(str(p.relative_to(ROOT)))
            except OSError:
                pass

    item["file_count"] = len(files)
    item["total_bytes"] = total_bytes
    item["sample_files"] = files[:20]

    return item


def git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )

        return {
            "available": True,
            "commit": commit,
            "status": status.splitlines(),
        }

    except Exception:
        return {
            "available": False
        }


def main() -> None:
    print("=" * 70)
    print("SatQueryAI C3 BASELINE")
    print("=" * 70)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git": git_info(),

        "raw_bigearthnet": {},
        "critical_processed_files": {},
        "critical_directories": {},
        "c2_checkpoint": {},
    }

    # ------------------------------------------------------------------
    # ORIGINAL RAW ARCHIVES
    # ------------------------------------------------------------------

    raw_files = [
        RAW_ROOT / "BigEarthNet-S1.tar.zst",
        RAW_ROOT / "BigEarthNet-S2.tar.zst",
        RAW_ROOT / "Reference_Maps.tar.zst",
    ]

    print("\n[1] RAW ARCHIVES")
    for path in raw_files:
        manifest["raw_bigearthnet"][path.name] = describe_file(
            path, hash_file=True
        )

    # ------------------------------------------------------------------
    # CRITICAL PROCESSED FILES
    # ------------------------------------------------------------------

    processed_files = [
        PROCESSED_ROOT / "training_annotations.parquet",
        PROCESSED_ROOT / "reference_grounding_split" / "reference_train.parquet",
        PROCESSED_ROOT / "reference_grounding_split" / "reference_val.parquet",
        PROCESSED_ROOT / "grounding_text_embeddings" /
        "grounding_text_embedding_shards",
    ]

    print("\n[2] CRITICAL PROCESSED PATHS")

    for path in processed_files:
        if path.is_file():
            manifest["critical_processed_files"][str(path)] = describe_file(
                path, hash_file=True
            )
        elif path.is_dir():
            manifest["critical_directories"][str(path)] = describe_directory(
                path
            )
        else:
            manifest["critical_processed_files"][str(path)] = {
                "path": str(path),
                "exists": False,
            }

    # ------------------------------------------------------------------
    # EXISTING CORE DIRECTORIES
    # ------------------------------------------------------------------

    directories = [
        PROCESSED_ROOT / "images_raw" / "BigEarthNet-S1",
        PROCESSED_ROOT / "images_raw" / "BigEarthNet-S2",
        PROCESSED_ROOT / "preprocessed",
        PROCESSED_ROOT / "dofa_features_download" / "dofa_features",
        PROCESSED_ROOT / "c2_diagnostic",
    ]

    print("\n[3] CORE DIRECTORIES")

    for path in directories:
        manifest["critical_directories"][str(path)] = describe_directory(path)

    # ------------------------------------------------------------------
    # C2 CHECKPOINT
    # ------------------------------------------------------------------

    print("\n[4] C2 CHECKPOINT")

    for path in CHECKPOINT_ROOT.glob("*"):
        if path.is_file():
            manifest["c2_checkpoint"][path.name] = describe_file(
                path, hash_file=True
            )

    # ------------------------------------------------------------------
    # DISK INFORMATION
    # ------------------------------------------------------------------

    try:
        usage = os.statvfs(ROOT.anchor)
        free_bytes = usage.f_frsize * usage.f_bavail

        manifest["disk"] = {
            "root": ROOT.anchor,
            "free_bytes": free_bytes,
            "free_gib": round(free_bytes / (1024 ** 3), 2),
        }
    except Exception:
        manifest["disk"] = {
            "available": False
        }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("BASELINE COMPLETE")
    print(f"Manifest: {MANIFEST_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()