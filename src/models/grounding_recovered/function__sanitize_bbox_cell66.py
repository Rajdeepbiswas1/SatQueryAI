def _sanitize_bbox(
    bbox
):

    bbox = bbox.float().clamp(
        0.0,
        1.0
    )

    x1 = torch.minimum(
        bbox[:, 0],
        bbox[:, 2]
    )

    y1 = torch.minimum(
        bbox[:, 1],
        bbox[:, 3]
    )

    x2 = torch.maximum(
        bbox[:, 0],
        bbox[:, 2]
    )

    y2 = torch.maximum(
        bbox[:, 1],
        bbox[:, 3]
    )

    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dim=-1
    )
