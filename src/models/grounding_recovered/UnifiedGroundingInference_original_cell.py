class UnifiedGroundingInference:

    def __init__(self, unified_model):
        self.model = unified_model
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        mode,
        s1,
        s2,
        text_embedding,
        point=None,
        query=None,
        reference=None,
        patch_id=None,
    ):

        mode = str(mode).strip().lower()

        if mode not in ("point", "reference"):
            raise ValueError(
                f"Unsupported mode '{mode}'. "
                "Expected 'point' or 'reference'."
            )

        # ---------------------------------------------------------------------
        # Input preparation
        # ---------------------------------------------------------------------

        s1 = _to_batched(s1, "s1")
        s2 = _to_batched(s2, "s2")
        text_embedding = _to_batched(
            text_embedding,
            "text_embedding"
        )

        s1 = _move_to_model_device(s1)
        s2 = _move_to_model_device(s2)
        text_embedding = _move_to_model_device(
            text_embedding
        )

        if mode == "point":

            if point is None:
                raise ValueError(
                    "Point grounding requires point=(x,y)."
                )

            point = _to_batched(point, "point")
            point = _move_to_model_device(point)

            native_output = self.model(
                mode="point",
                s1=s1,
                s2=s2,
                text_embedding=text_embedding,
                point=point,
            )

        else:

            native_output = self.model(
                mode="reference",
                s1=s1,
                s2=s2,
                text_embedding=text_embedding,
            )

        # ---------------------------------------------------------------------
        # Standardize evidence
        # ---------------------------------------------------------------------

        evidence = _extract_specialist_evidence(
            native_output
        )

        heatmap = evidence["heatmap"]
        bbox = evidence["bbox"]

        # ---------------------------------------------------------------------
        # Diagnostic confidence
        # ---------------------------------------------------------------------
        #
        # This is explicitly an evidence-concentration score.
        # It is NOT a calibrated probability of correctness.

        peak = evidence["peak_probability"]

        B, H, W = heatmap.shape

        x1 = torch.floor(
            bbox[:, 0] * (W - 1)
        ).long().clamp(0, W - 1)

        x2 = torch.ceil(
            bbox[:, 2] * (W - 1)
        ).long().clamp(0, W - 1)

        y1 = torch.floor(
            bbox[:, 1] * (H - 1)
        ).long().clamp(0, H - 1)

        y2 = torch.ceil(
            bbox[:, 3] * (H - 1)
        ).long().clamp(0, H - 1)

        concentration = []

        for i in range(B):
            region = heatmap[
                i,
                y1[i]:y2[i] + 1,
                x1[i]:x2[i] + 1
            ]

            concentration.append(
                region.sum()
            )

        concentration = torch.stack(concentration)

        evidence_confidence = (
            0.5 * peak +
            0.5 * concentration
        ).clamp(0.0, 1.0)

        # ---------------------------------------------------------------------
        # Standard result
        # ---------------------------------------------------------------------

        result = {
            "mode": mode,

            "bbox": bbox.cpu().numpy(),

            "heatmap": heatmap.cpu().numpy(),

            "evidence_confidence":
                evidence_confidence.cpu().numpy(),

            "peak_probability":
                peak.cpu().numpy(),

            "spatial_entropy":
                evidence["spatial_entropy"].cpu().numpy(),

            "bbox_source":
                evidence["bbox_source"],

            "query": query,

            "reference": reference,

            "patch_id": patch_id,

            "point": (
                point.cpu().numpy()
                if point is not None
                else None
            ),

            # Preserve the native specialist output.
            "native_output": native_output,
        }

        return result
