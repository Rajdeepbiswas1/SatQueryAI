def load_dofa(
    patch_id,
):

    global _dofa_path_cache
    global _dofa_data_cache

    if patch_id not in dofa_patch_index:
        raise RuntimeError(
            f"Patch not found in DOFA cache: {patch_id}"
        )

    path, row_idx = (
        dofa_patch_index[patch_id]
    )

    if _dofa_path_cache != path:

        _dofa_data_cache = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        _dofa_path_cache = path

    data = _dofa_data_cache

    s1 = data["s1"][
        row_idx
    ].float()

    s2 = data["s2"][
        row_idx
    ].float()

    return s1, s2
