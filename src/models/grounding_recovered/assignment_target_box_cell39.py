target_box = get_box(
    sample_batch
).float().to(
    DEVICE,
    non_blocking=True,
)
