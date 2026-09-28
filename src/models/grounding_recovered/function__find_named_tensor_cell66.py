def _find_named_tensor(
    output,
    names
):

    if not isinstance(output, dict):
        return None

    for name in names:

        value = output.get(name)

        if torch.is_tensor(value):
            return value

    return None
