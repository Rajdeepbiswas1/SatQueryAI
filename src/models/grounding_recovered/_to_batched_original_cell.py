def _to_batched(x, name):
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch Tensor.")

    if x.ndim == 1:
        return x.unsqueeze(0)

    if x.ndim == 2:
        return x.unsqueeze(0)

    return x
