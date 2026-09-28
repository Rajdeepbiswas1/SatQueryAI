class GroundingTextCache:
    """
    Verified text-cache interface.

    Index:
        source_row_index -> (shard_id, local_index)

    Expected fields:
        source_row_index
        text_embeddings

    Expected embedding size:
        768
    """

    def __init__(self, directory):

        self.directory = Path(directory)

        self.files = sorted(
            self.directory.glob(
                "grounding_text_shard_*.pt"
            )
        )

        if not self.files:
            raise RuntimeError(
                f"No grounding text shards found in:\n"
                f"{self.directory}"
            )

        self.index = {}
        self.shards = {}

        print()
        print("=" * 80)
        print("LOADING TEXT EMBEDDING CACHE")
        print("=" * 80)

        print(
            "Text embedding shards found:",
            len(self.files)
        )

        if len(self.files) != EXPECTED_TEXT_SHARDS:
            raise RuntimeError(
                "Expected "
                f"{EXPECTED_TEXT_SHARDS} "
                "text embedding shards, found "
                f"{len(self.files)}"
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
                    f"Unexpected text shard format:\n"
                    f"{filepath}"
                )

            required = [
                "source_row_index",
                "text_embeddings",
            ]

            for key in required:
                if key not in data:
                    raise RuntimeError(
                        f"Missing '{key}' in:\n"
                        f"{filepath}"
                    )

            source_indices = (
                data["source_row_index"]
            )

            embeddings = (
                data["text_embeddings"]
            )

            if not torch.is_tensor(
                source_indices
            ):
                raise RuntimeError(
                    f"'source_row_index' is not a tensor:\n"
                    f"{filepath}"
                )

            if not torch.is_tensor(
                embeddings
            ):
                raise RuntimeError(
                    f"'text_embeddings' is not a tensor:\n"
                    f"{filepath}"
                )

            if (
                embeddings.ndim != 2
                or embeddings.shape[1] != INPUT_DIM
            ):
                raise RuntimeError(
                    "Unexpected text embedding shape "
                    f"{tuple(embeddings.shape)} in:\n"
                    f"{filepath}"
                )

            if (
                len(source_indices)
                != len(embeddings)
            ):
                raise RuntimeError(
                    "source_row_index and "
                    "text_embeddings lengths differ in:\n"
                    f"{filepath}"
                )

            self.shards[shard_id] = data

            for local_index, row_index in enumerate(
                source_indices.tolist()
            ):

                row_index = int(row_index)

                if row_index in self.index:
                    raise RuntimeError(
                        "Duplicate source_row_index "
                        f"{row_index} across text shards."
                    )

                self.index[row_index] = (
                    shard_id,
                    local_index,
                )

        print(
            "Indexed text embeddings:",
            f"{len(self.index):,}"
        )

    def get(self, source_row_index):

        source_row_index = int(
            source_row_index
        )

        if source_row_index not in self.index:
            raise KeyError(
                "Text embedding missing for source row "
                f"{source_row_index}"
            )

        shard_id, local_index = (
            self.index[source_row_index]
        )

        embedding = (
            self.shards[shard_id]
            ["text_embeddings"]
            [local_index]
            .float()
        )

        return embedding
