class UnifiedGroundingModel(nn.Module):
    """
    Unified grounding interface.

    mode = "point"
        Uses the exact trained point-grounding specialist.

    mode = "reference"
        Uses the exact trained C3 reference-grounding specialist.

    This first integration deliberately does NOT retrain or
    modify either specialist.
    """

    def __init__(
        self,
        point_model,
        reference_model,
    ):
        super().__init__()

        self.point_model = point_model
        self.reference_model = reference_model

    def forward(
        self,
        mode,
        s1,
        s2,
        text_embedding,
        point=None,
    ):

        if mode == "point":

            if point is None:

                raise ValueError(
                    "Point mode requires point coordinates."
                )

            return self.point_model(
                s1,
                s2,
                text_embedding,
                point,
            )

        if mode == "reference":

            return self.reference_model(
                s1,
                s2,
                text_embedding,
            )

        raise ValueError(
            f"Unsupported grounding mode: {mode}"
        )
