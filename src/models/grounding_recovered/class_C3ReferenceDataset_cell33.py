class C3ReferenceDataset(
    Dataset
):

    def __init__(
        self,
        base_dataset,
        semantic_targets,
    ):

        self.base_dataset = (
            base_dataset
        )

        self.semantic_targets = (
            semantic_targets.float()
        )

        if len(
            self.base_dataset
        ) != self.semantic_targets.shape[0]:

            raise RuntimeError(
                "Base dataset and semantic "
                "target row counts differ."
            )

    def __len__(
        self
    ):

        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index
    ):

        item = dict(
            self.base_dataset[index]
        )

        semantic_target = (
            self.semantic_targets[
                index
            ]
            .clone()
        )

        if tuple(
            semantic_target.shape
        ) != (
            14,
            14
        ):

            raise RuntimeError(
                "Unexpected semantic target "
                f"shape at index {index}: "
                f"{tuple(semantic_target.shape)}"
            )

        item[
            "semantic_target"
        ] = semantic_target

        return item
