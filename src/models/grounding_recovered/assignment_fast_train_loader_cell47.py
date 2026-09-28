fast_train_loader = DataLoader(
    ram_train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    collate_fn=fast_collate_point,
)
