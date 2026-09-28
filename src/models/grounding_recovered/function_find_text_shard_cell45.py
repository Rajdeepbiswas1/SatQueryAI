def find_text_shard(index):

    for start, end, path in text_ranges:

        if start <= index < end:
            return (
                path,
                index - start,
            )

    raise IndexError(
        f"Text cache index out of range: {index}"
    )
