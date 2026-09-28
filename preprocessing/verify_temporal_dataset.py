import os
import random
import torch


PROJECT_ROOT = r"D:\SatQueryAI"

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "temporal_preprocessed",
)

EXPECTED_PAIRS = 2968
RANDOM_SAMPLES = 30
SEED = 42


def main():
    print("")
    print("=" * 70)
    print("FULL TEMPORAL DATASET VERIFICATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Find PT files
    # ---------------------------------------------------------

    pt_files = sorted(
        f
        for f in os.listdir(DATASET_ROOT)
        if f.lower().endswith(".pt")
    )

    pt_ids = {
        os.path.splitext(f)[0]
        for f in pt_files
    }

    print(
        f"Expected PT files: {EXPECTED_PAIRS}"
    )

    print(
        f"Found PT files: {len(pt_files)}"
    )

    if len(pt_files) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRS} PT files, "
            f"found {len(pt_files)}."
        )

    # ---------------------------------------------------------
    # Check duplicate IDs
    # ---------------------------------------------------------

    if len(pt_ids) != len(pt_files):
        raise RuntimeError(
            "Duplicate pair IDs detected."
        )

    # ---------------------------------------------------------
    # Random sample verification
    # ---------------------------------------------------------

    rng = random.Random(SEED)

    selected_files = rng.sample(
        pt_files,
        min(RANDOM_SAMPLES, len(pt_files)),
    )

    print("")
    print(
        f"Random samples checked: "
        f"{len(selected_files)}"
    )

    expected_t1_shape = (
        3,
        512,
        512,
    )

    expected_t2_shape = (
        3,
        512,
        512,
    )

    expected_label_shape = (
        512,
        512,
    )

    for filename in selected_files:

        path = os.path.join(
            DATASET_ROOT,
            filename,
        )

        sample = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        required_keys = {
            "pair_id",
            "t1",
            "t2",
            "label1",
            "label2",
            "change_mask",
            "transition_map",
            "class_names",
        }

        missing_keys = (
            required_keys
            - set(sample.keys())
        )

        if missing_keys:
            raise RuntimeError(
                f"{filename}: missing keys "
                f"{sorted(missing_keys)}"
            )

        # -----------------------------------------------------
        # Shapes
        # -----------------------------------------------------

        if tuple(sample["t1"].shape) != expected_t1_shape:
            raise RuntimeError(
                f"{filename}: invalid T1 shape "
                f"{tuple(sample['t1'].shape)}"
            )

        if tuple(sample["t2"].shape) != expected_t2_shape:
            raise RuntimeError(
                f"{filename}: invalid T2 shape "
                f"{tuple(sample['t2'].shape)}"
            )

        for key in [
            "label1",
            "label2",
            "change_mask",
            "transition_map",
        ]:
            if tuple(sample[key].shape) != expected_label_shape:
                raise RuntimeError(
                    f"{filename}: invalid {key} shape "
                    f"{tuple(sample[key].shape)}"
                )

        # -----------------------------------------------------
        # Dtypes
        # -----------------------------------------------------

        if sample["t1"].dtype != torch.float32:
            raise RuntimeError(
                f"{filename}: T1 dtype is "
                f"{sample['t1'].dtype}"
            )

        if sample["t2"].dtype != torch.float32:
            raise RuntimeError(
                f"{filename}: T2 dtype is "
                f"{sample['t2'].dtype}"
            )

        if sample["label1"].dtype != torch.int64:
            raise RuntimeError(
                f"{filename}: label1 dtype is "
                f"{sample['label1'].dtype}"
            )

        if sample["label2"].dtype != torch.int64:
            raise RuntimeError(
                f"{filename}: label2 dtype is "
                f"{sample['label2'].dtype}"
            )

        if sample["change_mask"].dtype != torch.uint8:
            raise RuntimeError(
                f"{filename}: change_mask dtype is "
                f"{sample['change_mask'].dtype}"
            )

        if sample["transition_map"].dtype != torch.int16:
            raise RuntimeError(
                f"{filename}: transition_map dtype is "
                f"{sample['transition_map'].dtype}"
            )

        # -----------------------------------------------------
        # Finite image values
        # -----------------------------------------------------

        if not torch.isfinite(
            sample["t1"]
        ).all():
            raise RuntimeError(
                f"{filename}: T1 contains "
                f"non-finite values."
            )

        if not torch.isfinite(
            sample["t2"]
        ).all():
            raise RuntimeError(
                f"{filename}: T2 contains "
                f"non-finite values."
            )

        # -----------------------------------------------------
        # Image range
        # -----------------------------------------------------

        if (
            sample["t1"].min().item() < 0
            or sample["t1"].max().item() > 1
        ):
            raise RuntimeError(
                f"{filename}: T1 outside [0,1]."
            )

        if (
            sample["t2"].min().item() < 0
            or sample["t2"].max().item() > 1
        ):
            raise RuntimeError(
                f"{filename}: T2 outside [0,1]."
            )

        # -----------------------------------------------------
        # Semantic class IDs
        # -----------------------------------------------------

        for key in [
            "label1",
            "label2",
        ]:
            label = sample[key]

            if not torch.all(
                (label >= 0)
                & (label <= 6)
            ):
                raise RuntimeError(
                    f"{filename}: {key} contains "
                    f"invalid class IDs."
                )

        # -----------------------------------------------------
        # Change mask validation
        # -----------------------------------------------------

        expected_change = (
            sample["label1"]
            != sample["label2"]
        ).to(torch.uint8)

        if not torch.equal(
            sample["change_mask"],
            expected_change,
        ):
            raise RuntimeError(
                f"{filename}: change_mask does "
                f"not match label1 != label2."
            )

        # -----------------------------------------------------
        # Transition map validation
        # -----------------------------------------------------

        expected_transition = (
            sample["label1"] * 10
            + sample["label2"]
        ).to(torch.int16)

        if not torch.equal(
            sample["transition_map"],
            expected_transition,
        ):
            raise RuntimeError(
                f"{filename}: transition_map "
                f"is inconsistent."
            )

        # -----------------------------------------------------
        # Pair ID validation
        # -----------------------------------------------------

        expected_id = os.path.splitext(
            filename
        )[0]

        if str(sample["pair_id"]) != expected_id:
            raise RuntimeError(
                f"{filename}: pair_id is "
                f"{sample['pair_id']}"
            )

    print("")
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    print(
        f"Total PT files: {len(pt_files)}"
    )

    print(
        f"Random samples validated: "
        f"{len(selected_files)}"
    )

    print(
        "All sampled tensors have valid "
        "shapes, dtypes, ranges, class IDs, "
        "change masks, and transition maps."
    )

    print("")
    print("SUCCESS")


if __name__ == "__main__":
    main()