def _normalize_heatmap(
    x
):

    if not torch.is_tensor(x):
        raise TypeError(
            "Heatmap must be a Tensor."
        )

    x = x.float()

    if x.ndim == 2:

        x = x.unsqueeze(0)

    elif x.ndim == 4:

        # Expected one spatial channel for a generic heatmap.
        x = x[:, 0]

    if x.ndim != 3:

        raise RuntimeError(
            f"Unexpected heatmap shape: "
            f"{tuple(x.shape)}"
        )

    sums = x.sum(
        dim=(-2, -1),
        keepdim=True
    )

    looks_like_probability = (
        torch.all(
            x >= 0
        ).item()
        and
        torch.all(
            x <= 1.0 + 1e-5
        ).item()
        and
        torch.all(
            sums > 0
        ).item()
    )

    if looks_like_probability:

        return (
            x
            /
            sums.clamp_min(1e-8)
        )

    return torch.softmax(
        x.flatten(1),
        dim=1
    ).reshape_as(x)
