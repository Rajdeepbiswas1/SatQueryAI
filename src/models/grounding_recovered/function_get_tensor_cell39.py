def get_tensor(batch, candidates):
    for key in candidates:
        if key in batch:
            value = batch[key]
            if torch.is_tensor(value):
                return value
    raise KeyError(
        f"None of these tensor keys were found: {candidates}"
    )
