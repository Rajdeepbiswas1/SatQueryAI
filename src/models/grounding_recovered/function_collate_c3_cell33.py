def collate_c3(
    batch
):

    semantic_targets = torch.stack(
        [
            item[
                "semantic_target"
            ]
            for item in batch
        ]
    )

    base_items = []

    for item in batch:

        item_copy = dict(
            item
        )

        del item_copy[
            "semantic_target"
        ]

        base_items.append(
            item_copy
        )

    output = collate_reference_batch(
        base_items
    )

    output[
        "semantic_target"
    ] = semantic_targets

    return output
