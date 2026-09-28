semantic_target = (
    semantic_target
    /
    semantic_target.sum(
        dim=(1, 2),
        keepdim=True,
    ).clamp_min(1e-8)
)
