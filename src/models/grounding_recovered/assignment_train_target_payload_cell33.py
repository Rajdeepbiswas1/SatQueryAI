train_target_payload = torch.load(
    TRAIN_TARGET_FILE,
    map_location="cpu",
    weights_only=False
)
