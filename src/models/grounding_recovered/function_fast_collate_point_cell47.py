def fast_collate_point(batch):

    return {
        "s1": torch.stack(
            [x["s1"] for x in batch],
            dim=0
        ),

        "s2": torch.stack(
            [x["s2"] for x in batch],
            dim=0
        ),

        "text": torch.stack(
            [x["text"] for x in batch],
            dim=0
        ),

        "point": torch.stack(
            [x["point"] for x in batch],
            dim=0
        ),

        "target_box": torch.stack(
            [x["target_box"] for x in batch],
            dim=0
        ),
    }
