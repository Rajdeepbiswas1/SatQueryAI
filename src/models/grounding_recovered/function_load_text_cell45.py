def load_text(
    grounding_row_index,
):

    global _text_path_cache
    global _text_data_cache

    path, row_idx = find_text_shard(
        int(grounding_row_index)
    )

    if _text_path_cache != path:

        _text_data_cache = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        _text_path_cache = path

    text = _text_data_cache[
        "text_embeddings"
    ][row_idx].float()

    return text
