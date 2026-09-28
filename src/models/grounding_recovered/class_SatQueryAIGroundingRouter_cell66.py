class SatQueryAIGroundingRouter:

    SUPPORTED_MODES = {
        "point",
        "reference"
    }

    def __init__(
        self,
        grounding_inference
    ):

        self.grounding_inference = (
            grounding_inference
        )

        self.model = (
            grounding_inference.model
        )

        self.model.eval()

    def _validate_request(
        self,
        mode,
        s1,
        s2,
        text_embedding,
        point=None,
    ):

        mode = str(
            mode
        ).strip().lower()

        if mode not in self.SUPPORTED_MODES:

            raise ValueError(
                f"Unsupported mode '{mode}'. "
                f"Supported modes: "
                f"{sorted(self.SUPPORTED_MODES)}"
            )

        if not torch.is_tensor(s1):
            raise TypeError(
                "s1 must be a Tensor."
            )

        if not torch.is_tensor(s2):
            raise TypeError(
                "s2 must be a Tensor."
            )

        if not torch.is_tensor(text_embedding):
            raise TypeError(
                "text_embedding must be a Tensor."
            )

        if mode == "point":

            if point is None:
                raise ValueError(
                    "Point grounding requires a point."
                )

            point_flat = point.detach().float().flatten()

            if point_flat.numel() != 2:
                raise ValueError(
                    "Point must contain exactly 2 coordinates."
                )

            if not torch.all(
                (point_flat >= 0.0)
                &
                (point_flat <= 1.0)
            ):
                raise ValueError(
                    "Point coordinates must lie in [0,1]."
                )

        return mode

    def _specialist_name(
        self,
        mode
    ):

        if mode == "point":

            return type(
                self.model.point_model
            ).__name__

        return type(
            self.model.reference_model
        ).__name__

    def dispatch(
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

        start_time = time.perf_counter()

        # ---------------------------------------------------------------------
        # Request validation
        # ---------------------------------------------------------------------

        mode = self._validate_request(
            mode=mode,
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
            point=point,
        )

        validation_end = time.perf_counter()

        specialist_name = (
            self._specialist_name(
                mode
            )
        )

        # ---------------------------------------------------------------------
        # Execute inference
        # ---------------------------------------------------------------------

        result = self.grounding_inference.predict(
            mode=mode,
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
            point=point,
            query=query,
            reference=reference,
            patch_id=patch_id,
        )

        inference_end = time.perf_counter()

        # ---------------------------------------------------------------------
        # Observable execution trace
        # ---------------------------------------------------------------------

        result["execution_trace"] = {

            "task":
                "grounding",

            "requested_mode":
                mode,

            "selected_specialist":
                specialist_name,

            "point_used":
                (
                    point is not None
                    if mode == "point"
                    else False
                ),

            "query_available":
                query is not None,

            "reference_available":
                reference is not None,

            "patch_id_available":
                patch_id is not None,

            "input_shapes": {

                "s1":
                    tuple(s1.shape),

                "s2":
                    tuple(s2.shape),

                "text_embedding":
                    tuple(
                        text_embedding.shape
                    ),

                "point":
                    (
                        tuple(point.shape)
                        if point is not None
                        else None
                    ),
            },

            "validation_time_ms":
                (
                    validation_end
                    -
                    start_time
                ) * 1000.0,

            "inference_time_ms":
                (
                    inference_end
                    -
                    validation_end
                ) * 1000.0,

            "total_time_ms":
                (
                    inference_end
                    -
                    start_time
                ) * 1000.0,

            "status":
                "success",
        }

        return result
