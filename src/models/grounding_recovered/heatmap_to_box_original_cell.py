def heatmap_to_box(
    heatmap,
):

    (
        center_x,
        center_y,
        std_x,
        std_y
    ) = heatmap_moments(
        heatmap
    )


    width = (
        std_x
        *
        math.sqrt(12.0)
    ).clamp(
        0.02,
        1.0
    )


    height = (
        std_y
        *
        math.sqrt(12.0)
    ).clamp(
        0.02,
        1.0
    )


    x1 = (
        center_x
        -
        width / 2.0
    ).clamp(
        0.0,
        1.0
    )


    y1 = (
        center_y
        -
        height / 2.0
    ).clamp(
        0.0,
        1.0
    )


    x2 = (
        center_x
        +
        width / 2.0
    ).clamp(
        0.0,
        1.0
    )


    y2 = (
        center_y
        +
        height / 2.0
    ).clamp(
        0.0,
        1.0
    )


    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dim=-1
    )
