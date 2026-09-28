def _prepare_s1_s2(
    x,
    name
):
    """
    [196,768] -> [1,196,768]
    [B,196,768] -> unchanged
    """

    if not torch.is_tensor(x):
        raise TypeError(
            f"{name} must be a Tensor."
        )

    if x.ndim == 2:

        if tuple(x.shape) != (196, 768):
            raise ValueError(
                f"{name} expected [196,768], "
                f"got {tuple(x.shape)}"
            )

        return x.unsqueeze(0)

    if x.ndim == 3:

        if x.shape[-1] != 768:
            raise ValueError(
                f"{name} final dimension must be 768, "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-2] != 196:
            raise ValueError(
                f"{name} expected 196 spatial tokens, "
                f"got {tuple(x.shape)}"
            )

        return x

    raise ValueError(
        f"{name} must have shape [196,768] "
        f"or [B,196,768]. Got {tuple(x.shape)}"
    )
