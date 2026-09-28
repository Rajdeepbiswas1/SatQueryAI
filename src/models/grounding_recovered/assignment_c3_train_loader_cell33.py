c3_train_loader = DataLoader(
    c3_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_c3,
    pin_memory=False,
)
