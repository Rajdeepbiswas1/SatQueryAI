class FastPointGroundingDataset(Dataset):

    def __init__(self, source_dataset, name="dataset"):
        self.name = name

        self.patch_features = {}
        self.patch_to_index = {}

        self.patch_indices = []
        self.text = []
        self.points = []
        self.target_boxes = []

        start = time.time()

        print()
        print("=" * 80)
        print(f"BUILDING RAM CACHE: {name}")
        print("=" * 80)

        for i in range(len(source_dataset)):

            sample = source_dataset[i]

            # ------------------------------------------------
            # Required fields
            # ------------------------------------------------
            patch_id = sample["patch_id"]

            s1 = sample["s1"]
            s2 = sample["s2"]
            text = sample["text"]
            point = sample["point"]
            target_box = sample["target_box"]

            # ------------------------------------------------
            # Cache each unique patch only once
            # ------------------------------------------------
            if patch_id not in self.patch_to_index:

                patch_idx = len(self.patch_features)

                self.patch_to_index[patch_id] = patch_idx

                # CPU half precision keeps RAM usage low.
                self.patch_features[patch_idx] = (
                    s1.detach().cpu().contiguous().half(),
                    s2.detach().cpu().contiguous().half(),
                )

            else:
                patch_idx = self.patch_to_index[patch_id]

            # ------------------------------------------------
            # Store row-level information
            # ------------------------------------------------
            self.patch_indices.append(patch_idx)

            self.text.append(
                text.detach().cpu().float().contiguous()
            )

            self.points.append(
                point.detach().cpu().float().contiguous()
            )

            self.target_boxes.append(
                target_box.detach().cpu().float().contiguous()
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------
            if (
                (i + 1) % 500 == 0
                or (i + 1) == len(source_dataset)
            ):
                elapsed = time.time() - start

                print(
                    f"{name}: "
                    f"{i+1:,}/{len(source_dataset):,} | "
                    f"unique patches: {len(self.patch_features):,} | "
                    f"time: {elapsed/60:.1f} min"
                )

        self.patch_indices = torch.tensor(
            self.patch_indices,
            dtype=torch.long
        )

        self.text = torch.stack(self.text)
        self.points = torch.stack(self.points)
        self.target_boxes = torch.stack(self.target_boxes)

        elapsed = time.time() - start

        print()
        print(f"{name} cache complete.")
        print(f"Rows: {len(self):,}")
        print(f"Unique patches: {len(self.patch_features):,}")
        print(f"Time: {elapsed/60:.2f} minutes")

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):

        patch_idx = int(self.patch_indices[idx])

        s1, s2 = self.patch_features[patch_idx]

        return {
            "s1": s1,
            "s2": s2,
            "text": self.text[idx],
            "point": self.points[idx],
            "target_box": self.target_boxes[idx],
        }
