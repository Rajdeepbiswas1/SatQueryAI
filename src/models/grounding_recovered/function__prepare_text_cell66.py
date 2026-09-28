def _prepare_text(
    x
):
    """
    [768] -> [1,768]
    [B,768] -> unchanged
    """

    if not torch.is_tensor(x):
        raise TypeError(
            "text_embedding must be a Tensor."
        )

    if x.ndim == 1:

        if x.shape[0] != 768:
            raise ValueError(
                f"Expected text embedding of size 768, "
                f"got {tuple(x.shape)}"
            )

        return x.unsqueeze(0)

    if x.ndim == 2:

        if x.shape[-1] != 768:
            raise ValueError(
                f"Text embedding final dimension must be 768, "
                f"got {tuple(x.shape)}"
            )

        return x

    raise ValueError(
        "text_embedding must have shape [768] "
        "or [B,768]."
    )
