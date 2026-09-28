from pathlib import Path
import torch

from agent.temporal_upload_adapter import TemporalUploadAdapter


PROJECT_ROOT = Path(r"D:\SatQueryAI")

T1_PATH = PROJECT_ROOT / "data" / "processed" / "temporal_raw" / "im1" / "00003.png"
T2_PATH = PROJECT_ROOT / "data" / "processed" / "temporal_raw" / "im2" / "00003.png"
SHARD_PATH = PROJECT_ROOT / "data" / "processed" / "temporal_features" / "shard_0000.pt"


def normalize_pair_ids(pair_ids):
    result = []

    for item in pair_ids:
        if isinstance(item, torch.Tensor):
            if item.numel() == 1:
                item = item.item()
            else:
                item = item.tolist()

        if isinstance(item, bytes):
            item = item.decode("utf-8")

        result.append(str(item))

    return result


def as_feature_tensor(x):
    x = torch.as_tensor(x).float()

    if x.ndim == 2:
        x = x.unsqueeze(0)

    if x.ndim != 3:
        raise AssertionError(f"Unexpected feature shape: {tuple(x.shape)}")

    return x


def compare(name, actual, expected, atol=1e-2, rtol=5e-3):
    actual = as_feature_tensor(actual)
    expected = as_feature_tensor(expected)

    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )

    abs_diff = torch.abs(actual - expected)
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()

    passed = torch.allclose(actual, expected, atol=atol, rtol=rtol)

    print(
        f"{name}: max_abs_error={max_err:.8f} "
        f"mean_abs_error={mean_err:.8f} "
        f"allclose={passed}"
    )

    if not passed:
        raise AssertionError(f"{name} exceeds tolerance")


def main():
    print("SATQUERYAI TEMPORAL UPLOAD ADAPTER REAL-DATA SMOKE TEST")
    print()

    for path in (T1_PATH, T2_PATH, SHARD_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    print("Input files: PASS")

    shard = torch.load(SHARD_PATH, map_location="cpu", weights_only=False)

    required_keys = {
        "pair_ids",
        "t1_features",
        "t2_features",
        "diff_features",
    }

    missing = required_keys - set(shard.keys())
    if missing:
        raise AssertionError(f"Missing shard keys: {sorted(missing)}")

    pair_ids = normalize_pair_ids(shard["pair_ids"])

    if "00003" not in pair_ids:
        raise AssertionError(
            f"SECOND pair 00003 not found in {SHARD_PATH}"
        )

    row = pair_ids.index("00003")
    print(f"SECOND pair: 00003")
    print(f"Feature shard row: {row}")

    expected_t1 = shard["t1_features"][row]
    expected_t2 = shard["t2_features"][row]
    expected_diff = shard["diff_features"][row]

    print(
        "Cached shapes:",
        tuple(torch.as_tensor(expected_t1).shape),
        tuple(torch.as_tensor(expected_t2).shape),
        tuple(torch.as_tensor(expected_diff).shape),
    )

    adapter = TemporalUploadAdapter(device="cpu")

    result = adapter.transform_pair(
        t1_path=T1_PATH,
        t2_path=T2_PATH,
    )

    actual_t1 = result["t1"]
    actual_t2 = result["t2"]
    actual_diff = result["diff"]

    print(
        "Adapter shapes:",
        tuple(actual_t1.shape),
        tuple(actual_t2.shape),
        tuple(actual_diff.shape),
    )

    expected_shape = (1, 196, 768)

    for name, tensor in (
        ("T1", actual_t1),
        ("T2", actual_t2),
        ("Diff", actual_diff),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise AssertionError(
                f"{name} output shape mismatch: "
                f"{tuple(tensor.shape)} != {expected_shape}"
            )

        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} contains non-finite values")

    print("Adapter shape/finite checks: PASS")

    compare("T1 features", actual_t1, expected_t1)
    compare("T2 features", actual_t2, expected_t2)
    compare("Diff features", actual_diff, expected_diff)

    calculated_diff = torch.abs(actual_t2 - actual_t1)

    if not torch.allclose(
        actual_diff,
        calculated_diff,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError("Adapter Diff != abs(T2 - T1)")

    print("Diff construction check: PASS")

    metadata = result.get("metadata", {})

    print()
    print("Backbone:", metadata.get("backbone"))
    print("Wavelengths:", metadata.get("wavelengths"))
    print("T1 input shape:", metadata.get("t1_input_shape"))
    print("T2 input shape:", metadata.get("t2_input_shape"))
    print("T1 feature shape:", metadata.get("t1_feature_shape"))
    print("T2 feature shape:", metadata.get("t2_feature_shape"))
    print("Diff feature shape:", metadata.get("diff_feature_shape"))
    print("Frozen:", metadata.get("frozen"))

    print()
    print("SECOND 00003 RAW -> DOFA -> CACHE EQUIVALENCE: PASS")
    print("TEMPORAL UPLOAD ADAPTER: VERIFIED")


if __name__ == "__main__":
    main()