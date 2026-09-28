c1_model = GroundingHeatmapModel(
    input_dim=768,
    hidden_dim=256,
    grid_size=14
).to(device)
