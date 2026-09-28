def _heatmap_moment_box_generic(
    prob
):

    B, H, W = prob.shape

    ys = torch.linspace(
        0.0,
        1.0,
        H,
        device=prob.device,
        dtype=prob.dtype
    )

    xs = torch.linspace(
        0.0,
        1.0,
        W,
        device=prob.device,
        dtype=prob.dtype
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    cx = (
        prob * xx
    ).sum(
        dim=(-2, -1)
    )

    cy = (
        prob * yy
    ).sum(
        dim=(-2, -1)
    )

    var_x = (
        prob
        *
        (
            xx
            -
            cx[:, None, None]
        ).pow(2)
    ).sum(
        dim=(-2, -1)
    )

    var_y = (
        prob
        *
        (
            yy
            -
            cy[:, None, None]
        ).pow(2)
    ).sum(
        dim=(-2, -1)
    )

    std_x = torch.sqrt(
        var_x.clamp_min(1e-8)
    )

    std_y = torch.sqrt(
        var_y.clamp_min(1e-8)
    )

    return torch.stack(
        [
            (
                cx - 2.0 * std_x
            ).clamp(0.0, 1.0),

            (
                cy - 2.0 * std_y
            ).clamp(0.0, 1.0),

            (
                cx + 2.0 * std_x
            ).clamp(0.0, 1.0),

            (
                cy + 2.0 * std_y
            ).clamp(0.0, 1.0),
        ],
        dim=-1
    )
