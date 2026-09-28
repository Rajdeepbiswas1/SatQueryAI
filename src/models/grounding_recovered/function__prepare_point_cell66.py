def _prepare_point(
    x
):
    """
    [2] -> [1,2]
    [B,2] -> unchanged

    IMPORTANT:
        A [B,2] point is never unsqueezed.
    """

    if not torch.is_tensor(x):
        raise TypeError(
            "point must be a Tensor."
        )

    if x.ndim == 1:

        if x.numel() != 2:
            raise ValueError(
                f"Point must contain exactly 2 values, "
                f"got {tuple(x.shape)}"
            )

        return x.reshape(
            1,
            2
        )

    if x.ndim == 2:

        if x.shape[-1] != 2:
            raise ValueError(
                f"Point expected [B,2], "
                f"got {tuple(x.shape)}"
            )

        return x

    raise ValueError(
        f"Point must have shape [2] "
        f"or [B,2]. Got {tuple(x.shape)}"
    )
