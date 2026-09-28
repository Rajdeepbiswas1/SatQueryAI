class UnifiedGroundingInference:

    def __init__(
        self,
        unified_model
    ):

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

        mode = str(
            mode
        ).strip().lower()

        if mode not in (
            "point",
            "reference"
        ):
            raise ValueError(
                f"Unsupported grounding mode: {mode}"
            )

        # ---------------------------------------------------------------------
        # Prepare inputs
        # ---------------------------------------------------------------------

        s1 = _prepare_s1_s2(
            s1,
            "s1"
        )

        s2 = _prepare_s1_s2(
            s2,
            "s2"
        )

        text_embedding = _prepare_text(
            text_embedding
        )

        if mode == "point":

            if point is None:
                raise ValueError(
                    "Point grounding requires point."
                )

            point = _prepare_point(
                point
            )

        # ---------------------------------------------------------------------
        # Move to model device
        # ---------------------------------------------------------------------

        device = next(
            self.model.parameters()
        ).device

        s1 = s1.to(device)
        s2 = s2.to(device)
        text_embedding = text_embedding.to(
            device
        )

        if point is not None:
            point = point.to(device)

        # ---------------------------------------------------------------------
        # Shape validation
        # ---------------------------------------------------------------------

        if s1.ndim != 3:
            raise RuntimeError(
                f"s1 must be [B,196,768], got {tuple(s1.shape)}"
            )

        if s2.ndim != 3:
            raise RuntimeError(
                f"s2 must be [B,196,768], got {tuple(s2.shape)}"
            )

        if text_embedding.ndim != 2:
            raise RuntimeError(
                f"text_embedding must be [B,768], "
                f"got {tuple(text_embedding.shape)}"
            )

        if s1.shape[0] != s2.shape[0]:
            raise RuntimeError(
                "S1 and S2 batch sizes do not match."
            )

        if text_embedding.shape[0] != s1.shape[0]:
            raise RuntimeError(
                "Text and visual batch sizes do not match."
            )

        if mode == "point":

            if point.ndim != 2:
                raise RuntimeError(
                    f"point must be [B,2], got {tuple(point.shape)}"
                )

            if point.shape[0] != s1.shape[0]:
                raise RuntimeError(
                    "Point and visual batch sizes do not match."
                )

        # ---------------------------------------------------------------------
        # Specialist dispatch
        # ---------------------------------------------------------------------

        if mode == "point":

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

        evidence = _extract_evidence(
            native_output
        )

        heatmap = evidence[
            "heatmap"
        ]

        bbox = evidence[
            "bbox"
        ]

        peak = evidence[
            "peak_probability"
        ]

        # ---------------------------------------------------------------------
        # Evidence concentration
        # ---------------------------------------------------------------------

        B, H, W = heatmap.shape

        x1 = torch.floor(
            bbox[:, 0] * (W - 1)
        ).long().clamp(
            0,
            W - 1
        )

        x2 = torch.ceil(
            bbox[:, 2] * (W - 1)
        ).long().clamp(
            0,
            W - 1
        )

        y1 = torch.floor(
            bbox[:, 1] * (H - 1)
        ).long().clamp(
            0,
            H - 1
        )

        y2 = torch.ceil(
            bbox[:, 3] * (H - 1)
        ).long().clamp(
            0,
            H - 1
        )

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

        concentration = torch.stack(
            concentration
        )

        # ---------------------------------------------------------------------
        # Diagnostic evidence score
        # ---------------------------------------------------------------------
        #
        # IMPORTANT:
        # This is NOT a calibrated probability of correctness.
        # ---------------------------------------------------------------------

        evidence_confidence = (
            0.5 * peak
            +
            0.5 * concentration
        ).clamp(
            0.0,
            1.0
        )

        # ---------------------------------------------------------------------
        # Standard result
        # ---------------------------------------------------------------------

        return {
            "mode": mode,

            "bbox":
                bbox.detach()
                .cpu()
                .numpy(),

            "heatmap":
                heatmap.detach()
                .cpu()
                .numpy(),

            "evidence_confidence":
                evidence_confidence.detach()
                .cpu()
                .numpy(),

            "peak_probability":
                peak.detach()
                .cpu()
                .numpy(),

            "spatial_entropy":
                evidence[
                    "spatial_entropy"
                ]
                .detach()
                .cpu()
                .numpy(),

            "bbox_source":
                evidence[
                    "bbox_source"
                ],

            "query": query,

            "reference": reference,

            "patch_id": patch_id,

            "point": (
                point.detach()
                .cpu()
                .numpy()
                if point is not None
                else None
            ),

            "native_output":
                native_output,
        }
