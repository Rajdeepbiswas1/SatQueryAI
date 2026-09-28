class PointGroundingDataset(
    Dataset
):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        patch_id = str(
            row["patch_id"]
        )

        s1, s2 = load_dofa(
            patch_id
        )

        text = load_text(
            row["grounding_row_index"]
        )

        point = torch.tensor(
            row["point_xy"],
            dtype=torch.float32,
        )

        target_box = torch.tensor(
            row["target_box"],
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # Alignment checks
        # ----------------------------------------------------

        if not torch.isfinite(s1).all():
            raise RuntimeError(
                f"Non-finite S1 for {patch_id}"
            )

        if not torch.isfinite(s2).all():
            raise RuntimeError(
                f"Non-finite S2 for {patch_id}"
            )

        if not torch.isfinite(text).all():
            raise RuntimeError(
                f"Non-finite text at row "
                f"{row['grounding_row_index']}"
            )

        if not torch.isfinite(point).all():
            raise RuntimeError(
                "Non-finite point"
            )

        if not torch.isfinite(target_box).all():
            raise RuntimeError(
                "Non-finite target box"
            )

        # Point must lie inside target box.
        x, y = point.tolist()

        x1, y1, x2, y2 = (
            target_box.tolist()
        )

        if not (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):
            raise RuntimeError(
                f"Point outside target box at "
                f"row {idx}"
            )

        return {
            "s1": s1,
            "s2": s2,
            "text": text,
            "point": point,
            "target_box": target_box,
            "patch_id": patch_id,
            "grounding_row_index":
                int(row["grounding_row_index"]),
        }
