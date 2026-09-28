def collate_reference_batch(
    batch
):

    return {

        "s1": torch.stack(
            [
                item["s1"]
                for item in batch
            ]
        ),

        "s2": torch.stack(
            [
                item["s2"]
                for item in batch
            ]
        ),

        "text_embedding": torch.stack(
            [
                item["text_embedding"]
                for item in batch
            ]
        ),

        "target_box": torch.stack(
            [
                item["target_box"]
                for item in batch
            ]
        ),

        "patch_id": [
            item["patch_id"]
            for item in batch
        ],

        "row_index": [
            item["row_index"]
            for item in batch
        ],

        "category": [
            item["category"]
            for item in batch
        ],

        "phrase": [
            item["phrase"]
            for item in batch
        ],
    }
