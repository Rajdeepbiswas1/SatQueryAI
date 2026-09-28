class DofaShardCache:
    """
    Verified grounding-cache interface.

    Index:
        patch_id -> (shard_id, local_index)

    Expected shard fields:
        patch_ids
        s1
        s2
    """

    def __init__(self, directory):
        self.directory = Path(directory)

        self.files = sorted(
            self.directory.glob("*.pt")
        )

        if not self.files:
            raise RuntimeError(
                f"No DOFA shards found in:\n"
                f"{self.directory}"
            )

        self.index = {}
        self.shards = {}

        print()
        print("=" * 80)
        print("LOADING DOFA CACHE")
        print("=" * 80)

        print(
            "DOFA shards found:",
            len(self.files)
        )

        for shard_id, filepath in enumerate(
            self.files
        ):
            data = torch.load(
                filepath,
                map_location="cpu",
                weights_only=False,
            )

            if not isinstance(data, dict):
                raise RuntimeError(
                    f"Unexpected DOFA shard format:\n"
                    f"{filepath}"
                )

            required = [
                "patch_ids",
                "s1",
                "s2",
            ]

            for key in required:
                if key not in data:
                    raise RuntimeError(
                        f"Missing '{key}' in:\n"
                        f"{filepath}"
                    )

            s1 = data["s1"]
            s2 = data["s2"]
            patch_ids = data["patch_ids"]

            if not torch.is_tensor(s1):
                raise RuntimeError(
                    f"'s1' is not a tensor:\n{filepath}"
                )

            if not torch.is_tensor(s2):
                raise RuntimeError(
                    f"'s2' is not a tensor:\n{filepath}"
                )

            if s1.ndim != 3:
                raise RuntimeError(
                    f"Unexpected s1 shape "
                    f"{tuple(s1.shape)} in:\n{filepath}"
                )

            if s2.ndim != 3:
                raise RuntimeError(
                    f"Unexpected s2 shape "
                    f"{tuple(s2.shape)} in:\n{filepath}"
                )

            if s1.shape[-2:] != (
                VISUAL_TOKENS,
                INPUT_DIM,
            ):
                raise RuntimeError(
                    f"Unexpected s1 shape "
                    f"{tuple(s1.shape)} in:\n{filepath}"
                )

            if s2.shape[-2:] != (
                VISUAL_TOKENS,
                INPUT_DIM,
            ):
                raise RuntimeError(
                    f"Unexpected s2 shape "
                    f"{tuple(s2.shape)} in:\n{filepath}"
                )

            if len(patch_ids) != s1.shape[0]:
                raise RuntimeError(
                    "patch_ids and s1 length differ in:\n"
                    f"{filepath}"
                )

            if len(patch_ids) != s2.shape[0]:
                raise RuntimeError(
                    "patch_ids and s2 length differ in:\n"
                    f"{filepath}"
                )

            self.shards[shard_id] = data

            for local_index, patch_id in enumerate(
                patch_ids
            ):
                if torch.is_tensor(patch_id):
                    patch_id = patch_id.item()

                patch_id = str(patch_id)

                if patch_id in self.index:
                    raise RuntimeError(
                        "Duplicate patch ID in DOFA cache:\n"
                        f"{patch_id}"
                    )

                self.index[patch_id] = (
                    shard_id,
                    local_index,
                )

        print(
            "Indexed patch IDs:",
            f"{len(self.index):,}"
        )

    def get(self, patch_id):

        patch_id = str(patch_id)

        if patch_id not in self.index:
            raise KeyError(
                "Patch not found in DOFA cache:\n"
                f"{patch_id}"
            )

        shard_id, local_index = (
            self.index[patch_id]
        )

        shard = self.shards[shard_id]

        s1 = shard["s1"][local_index].float()
        s2 = shard["s2"][local_index].float()

        return s1, s2
