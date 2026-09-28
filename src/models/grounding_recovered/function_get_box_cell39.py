def get_box(batch):
    return get_tensor(
        batch,
        [
            "target_box",
            "box",
            "target_boxes",
            "bbox",
        ]
    )
