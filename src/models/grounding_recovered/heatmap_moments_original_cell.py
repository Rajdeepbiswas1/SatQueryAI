def heatmap_moments(
    heatmap,
):

    B, H, W = heatmap.shape

    device_ = heatmap.device
    dtype_ = heatmap.dtype


    xs = (
        torch.arange(
            W,
            device=device_,
            dtype=dtype_
        )
        + 0.5
    ) / W


    ys = (
        torch.arange(
            H,
            device=device_,
            dtype=dtype_
        )
        + 0.5
    ) / H


    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )


    xx = xx.reshape(
        1,
        -1
    )

    yy = yy.reshape(
        1,
        -1
    )


    p = heatmap.reshape(
        B,
        -1
    )


    # --------------------------------------------------------
    # FIRST MOMENT
    # --------------------------------------------------------

    center_x = (
        p * xx
    ).sum(
        dim=1
    )

    center_y = (
        p * yy
    ).sum(
        dim=1
    )


    # --------------------------------------------------------
    # SECOND MOMENT
    # --------------------------------------------------------

    variance_x = (
        p
        *
        (
            xx
            -
            center_x.unsqueeze(1)
        ).pow(2)
    ).sum(
        dim=1
    )


    variance_y = (
        p
        *
        (
            yy
            -
            center_y.unsqueeze(1)
        ).pow(2)
    ).sum(
        dim=1
    )


    std_x = torch.sqrt(
        variance_x
        +
        1e-8
    )

    std_y = torch.sqrt(
        variance_y
        +
        1e-8
    )


    return (
        center_x,
        center_y,
        std_x,
        std_y
    )
