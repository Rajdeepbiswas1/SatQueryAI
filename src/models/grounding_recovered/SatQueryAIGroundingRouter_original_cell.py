class SatQueryAIGroundingRouter:
    """
    Agent-facing orchestration layer for grounding.

    Supported modes:
        point
        reference

    The router performs:
        1. Request validation
        2. Specialist selection
        3. Unified inference
        4. Evidence standardization
        5. Observable execution tracing
    """

    SUPPORTED_MODES = {"point", "reference"}

    def __init__(self, grounding_inference):
        self.grounding_inference = grounding_inference

        self.model = grounding_inference.model
        self.model.eval()

    def _validate_request(
        self,
        mode,
        s1,
        s2,
        text_embedding,
        point=None,
    ):

        mode = str(mode).strip().lower()

        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported grounding mode '{mode}'. "
                f"Supported modes: {sorted(self.SUPPORTED_MODES)}"
            )

        if not torch.is_tensor(s1):
            raise TypeError("s1 must be a torch Tensor.")

        if not torch.is_tensor(s2):
            raise TypeError("s2 must be a torch Tensor.")

        if not torch.is_tensor(text_embedding):
            raise TypeError(
                "text_embedding must be a torch Tensor."
            )

        if mode == "point":

            if point is None:
                raise ValueError(
                    "Point grounding requires a point=(x,y)."
                )

            if not torch.is_tensor(point):
                raise TypeError(
                    "point must be a torch Tensor."
                )

            if point.numel() != 2:
                raise ValueError(
                    f"Point must contain exactly two coordinates. "
                    f"Received shape: {tuple(point.shape)}"
                )

            point_check = point.detach().float().flatten()

            if not torch.all(
                (point_check >= 0.0) &
                (point_check <= 1.0)
            ):
                raise ValueError(
                    "Point coordinates must be normalized to [0,1]."
                )

        return mode

    def _specialist_name(self, mode):

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
        # Validate
        # ---------------------------------------------------------------------

        mode = self._validate_request(
            mode=mode,
            s1=s1,
            s2=s2,
            text_embedding=text_embedding,
            point=point,
        )

        validation_time = time.perf_counter()

        specialist = self._specialist_name(mode)

        # ---------------------------------------------------------------------
        # Execute specialist
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

        inference_time = time.perf_counter()

        # ---------------------------------------------------------------------
        # Observable execution trace
        # ---------------------------------------------------------------------

        execution_trace = {
            "task": "grounding",

            "requested_mode": mode,

            "selected_specialist": specialist,

            "point_used": (
                point is not None
                if mode == "point"
                else False
            ),

            "query_available": query is not None,

            "reference_available": reference is not None,

            "patch_id_available": patch_id is not None,

            "input_shapes": {
                "s1": tuple(s1.shape),
                "s2": tuple(s2.shape),
                "text_embedding": tuple(
                    text_embedding.shape
                ),
                "point": (
                    tuple(point.shape)
                    if point is not None
                    else None
                ),
            },

            "validation_time_ms": (
                validation_time - start_time
            ) * 1000.0,

            "inference_time_ms": (
                inference_time - validation_time
            ) * 1000.0,

            "total_time_ms": (
                inference_time - start_time
            ) * 1000.0,

            "status": "success",
        }

        result["execution_trace"] = execution_trace

        return result
