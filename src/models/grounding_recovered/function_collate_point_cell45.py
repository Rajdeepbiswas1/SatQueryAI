def collate_point(
    batch
):

    return {

        "s1": torch.stack(
            [x["s1"] for x in batch]
        ),

        "s2": torch.stack(
            [x["s2"] for x in batch]
        ),

        "text": torch.stack(
            [x["text"] for x in batch]
        ),

        "point": torch.stack(
            [x["point"] for x in batch]
        ),

        "target_box": torch.stack(
            [x["target_box"] for x in batch]
        ),

        "patch_id": [
            x["patch_id"]
            for x in batch
        ],

        "grounding_row_index": [
            x["grounding_row_index"]
            for x in batch
        ],
    }
