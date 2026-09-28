from typing import Any, Dict


class SatQueryAICrossModalVQARouter:
    """
    Dispatcher for the cross-modal optical + SAR VQA capability.

    Selection rule:
        options is None -> Cross-Modal Binary VQA
        options present -> Cross-Modal MCQ VQA

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

        del kwargs

        if analysis.task != "cross_modal":
            raise ValueError(
                "SatQueryAICrossModalVQARouter received "
                f"task={analysis.task!r}; expected 'cross_modal'."
            )

        if s1 is None:
            raise ValueError(
                "Cross-modal VQA requires s1."
            )

        if s2 is None:
            raise ValueError(
                "Cross-modal VQA requires s2."
            )

        if text_embedding is None:
            raise ValueError(
                "Cross-modal VQA requires text_embedding."
            )

        common = {
            "query": query,
            "analysis": analysis,
            "s1": s1,
            "s2": s2,
            "text_embedding": text_embedding,
        }

        # --------------------------------------------------------------
        # Cross-Modal Binary VQA
        # --------------------------------------------------------------
        if options is None:

            result = self.binary_tool(
                **common
            )

            if not isinstance(
                result,
                dict,
            ):
                raise RuntimeError(
                    "Cross-modal Binary VQA tool returned "
                    "a non-dictionary result."
                )

            result = dict(result)

            result["cross_modal_vqa_route"] = "binary"
            result["specialist_selected"] = (
                "CrossModalBinaryVQA"
            )

            return result

        # --------------------------------------------------------------
        # Cross-Modal MCQ VQA
        # --------------------------------------------------------------
        result = self.mcq_tool(
            **common,
            options=options,
        )

        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                "Cross-modal MCQ VQA tool returned "
                "a non-dictionary result."
            )

        result = dict(result)

        result["cross_modal_vqa_route"] = "mcq"
        result["specialist_selected"] = (
            "CrossModalMCQVQA"
        )

        return result