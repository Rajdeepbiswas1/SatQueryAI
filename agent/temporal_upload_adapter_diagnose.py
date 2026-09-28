from pathlib import Path
import torch

from agent.temporal_upload_adapter import TemporalUploadAdapter


PROJECT_ROOT = Path(r"D:\SatQueryAI")

T1_RAW = PROJECT_ROOT / "data" / "processed" / "temporal_raw" / "im1" / "00003.png"
T2_RAW = PROJECT_ROOT / "data" / "processed" / "temporal_raw" / "im2" / "00003.png"

PREPROCESSED = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_preprocessed"
    / "00003.pt"
)

STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_rgb_stats.pt"
)


def describe(name, value):
    print(f"\n{name}")

    if torch.is_tensor(value):
        x = value.detach().float().cpu()
        print("  shape:", tuple(value.shape))
        print("  dtype:", value.dtype)
        print("  min:", float(x.min()))
        print("  max:", float(x.max()))
        print("  mean:", float(x.mean()))
        print("  std:", float(x.std()))

    elif isinstance(value, dict):
        print("  dict keys:", sorted(value.keys()))

    else:
        print("  type:", type(value).__name__)
        print("  value:", value)


def recursively_describe(prefix, value, depth=0):
    if depth > 2:
        return

    if torch.is_tensor(value):
        describe(prefix, value)
        return

    if isinstance(value, dict):
        print(f"\n{prefix} -> dict keys: {sorted(value.keys())}")

        for key, item in value.items():
            recursively_describe(
                f"{prefix}.{key}",
                item,
                depth + 1,
            )


def find_image_like_tensors(obj, prefix="root"):
    found = []

    if torch.is_tensor(obj):
        shape = tuple(obj.shape)

        if shape in {
            (3, 512, 512),
            (512, 512, 3),
            (3, 224, 224),
            (224, 224, 3),
        }:
            found.append((prefix, obj))

        return found

    if isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(
                find_image_like_tensors(
                    value,
                    f"{prefix}.{key}",
                )
            )

    return found


def compare_tensor(name, a, b):
    a = torch.as_tensor(a).float().cpu()
    b = torch.as_tensor(b).float().cpu()

    if a.shape != b.shape:
        print(
            f"{name}: shape mismatch "
            f"{tuple(a.shape)} vs {tuple(b.shape)}"
        )
        return

    diff = torch.abs(a - b)

    print(
        f"{name}: "
        f"max_abs={float(diff.max()):.8f}, "
        f"mean_abs={float(diff.mean()):.8f}, "
        f"allclose={torch.allclose(a, b, atol=1e-5, rtol=1e-5)}"
    )


def main():
    print("SATQUERYAI TEMPORAL UPLOAD ADAPTER DIAGNOSTIC")
    print("=" * 60)

    for path in (
        T1_RAW,
        T2_RAW,
        PREPROCESSED,
        STATS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    print("\nRequired files: PASS")

    print("\nLoading temporal_rgb_stats.pt")
    stats = torch.load(
        STATS_PATH,
        map_location="cpu",
        weights_only=False,
    )

    recursively_describe("stats", stats)

    print("\nLoading temporal_preprocessed/00003.pt")
    cached = torch.load(
        PREPROCESSED,
        map_location="cpu",
        weights_only=False,
    )

    recursively_describe("cached", cached)

    print("\nImage-like tensors found in preprocessed cache")
    candidates = find_image_like_tensors(cached)

    if not candidates:
        print("  NONE FOUND")
    else:
        for name, tensor in candidates:
            describe(name, tensor)

    print("\nInitializing adapter")
    adapter = TemporalUploadAdapter(device="cpu")

    print("\nAdapter loaded statistics")
    print("  T1 mean:", adapter.t1_mean.flatten().tolist())
    print("  T1 std :", adapter.t1_std.flatten().tolist())
    print("  T2 mean:", adapter.t2_mean.flatten().tolist())
    print("  T2 std :", adapter.t2_std.flatten().tolist())

    print("\nRaw image loading")
    raw_t1 = adapter._load_rgb_image(T1_RAW)
    raw_t2 = adapter._load_rgb_image(T2_RAW)

    describe("adapter raw T1", raw_t1)
    describe("adapter raw T2", raw_t2)

    print("\nAdapter normalized/resized inputs")

    prepared_t1 = adapter._prepare_image(
        raw_t1,
        adapter.t1_mean,
        adapter.t1_std,
    )

    prepared_t2 = adapter._prepare_image(
        raw_t2,
        adapter.t2_mean,
        adapter.t2_std,
    )

    describe("adapter prepared T1", prepared_t1)
    describe("adapter prepared T2", prepared_t2)

    print("\nSearching cached tensors for comparable inputs")

    for name, tensor in candidates:
        shape = tuple(tensor.shape)

        if shape in {
            (3, 512, 512),
            (512, 512, 3),
        }:
            print(f"\nCandidate: {name}")

            x = tensor.float()

            if tuple(x.shape) == (512, 512, 3):
                x = x.permute(2, 0, 1)

            # Compare against raw [0,1] tensor.
            compare_tensor(
                f"{name} vs raw T1",
                x,
                raw_t1,
            )

        elif shape in {
            (3, 224, 224),
            (224, 224, 3),
        }:
            print(f"\nCandidate: {name}")

            x = tensor.float()

            if tuple(x.shape) == (224, 224, 3):
                x = x.permute(2, 0, 1)

            compare_tensor(
                f"{name} vs prepared T1",
                x,
                prepared_t1.squeeze(0),
            )

    print("\nDiagnostic complete.")
    print(
        "DO NOT modify the adapter from this output alone; "
        "use the reported cache structure to identify the exact mismatch."
    )


if __name__ == "__main__":
    main()