from typing import Any, Dict


class SatQueryAISingleImageVQARouter:
    """
    Dispatcher for the single-image VQA capability.

    Selection rule:
        no options -> Binary VQA
        options    -> option-aware MCQ VQA

    The underlying specialist models remain independent and frozen.
    """

    def __init__(
        self,
        binary_tool,
        mcq_tool,
    ) -> None:

        if binary_tool is None:
            raise ValueError(
                "binary_tool is required."
            )

        if mcq_tool is None:
            raise ValueError(
                "mcq_tool is required."
            )

        self.binary_tool = binary_tool
        self.mcq_tool = mcq_tool

    def __call__(
        self,
        query: str,
        analysis,
        s1=None,
        s2=None,
        text_embedding=None,
        options=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        if analysis.task != "single_image_vqa":
            raise ValueError(
                "SatQueryAISingleImageVQARouter received "
                f"task={analysis.task!r}; expected 'single_image_vqa'."
            )

        common = {
            "query": query,
            "analysis": analysis,
            "s1": s1,
            "s2": s2,
            "text_embedding": text_embedding,
        }

        # --------------------------------------------------------------
        # Binary VQA
        # --------------------------------------------------------------
        if options is None:

            result = self.binary_tool(
                **common
            )

            if not isinstance(result, dict):
                raise RuntimeError(
                    "Binary VQA tool returned a non-dictionary result."
                )

            result = dict(result)

            result["vqa_route"] = "binary"
            result["specialist_selected"] = (
                "CachedBinaryVQA"
            )

            return result

        # --------------------------------------------------------------
        # MCQ VQA
        # --------------------------------------------------------------
        result = self.mcq_tool(
            **common,
            options=options,
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "MCQ VQA tool returned a non-dictionary result."
            )

        result = dict(result)

        result["vqa_route"] = "mcq"
        result["specialist_selected"] = (
            "MCQOptionAwareVQA"
        )

        return result