# =============================================================================
# SATQUERYAI — TEMPORAL AGENT APPLICATION SERVICE
# =============================================================================
#
# Bridges the verified TemporalRuntimeStore with the generic SatQueryAIAgent.
#
# This file:
#   - resolves a held-out Test question_id
#   - obtains t1/t2/diff/question embedding
#   - invokes the existing Agent unchanged
#   - formats the application response
#
# This file does NOT:
#   - modify the Agent core
#   - modify the Temporal V3 model
#   - modify checkpoints
#   - modify caches
# =============================================================================

from __future__ import annotations

from typing import Any

from agent.response_formatter import (
    format_agent_response,
)
from agent.temporal_runtime_store import (
    TemporalRuntimeStore,
)


class TemporalAgentService:
    """
    Application-level temporal execution service.
    """

    def __init__(
        self,
        agent,
        store: TemporalRuntimeStore | None = None,
    ) -> None:

        if agent is None:
            raise ValueError(
                "agent must not be None."
            )

        self.agent = agent

        self.store = (
            store
            if store is not None
            else TemporalRuntimeStore()
        )

    def execute_test_question(
        self,
        question_id: int,
    ) -> dict[str, Any]:
        """
        Execute one verified temporal Test question through the real Agent.
        """

        data = self.store.get(
            int(question_id)
        )

        raw_result = self.agent.execute(
            query=data["question"],
            t1=data["t1"],
            t2=data["t2"],
            diff=data["diff"],
            text_embedding=data[
                "text_embedding"
            ],
        )

        response = format_agent_response(
            raw_result
        )

        # Application metadata from the verified manifest.
        response["temporal_input"] = {
            "question_id": data["question_id"],
            "pair_id": data["pair_id"],
            "file_name": data["file_name"],
            "question_type": data["question_type"],
        }

        return response


# =============================================================================
# DIRECT TEST
# =============================================================================

if __name__ == "__main__":

    from agent.bootstrap import (
        create_runtime_agent,
    )

    print("=" * 80)
    print(
        "SATQUERYAI — TEMPORAL AGENT SERVICE TEST"
    )
    print("=" * 80)

    runtime_agent = create_runtime_agent()[0]

    service = TemporalAgentService(
        runtime_agent
    )

    response = service.execute_test_question(
        question_id=0
    )

    print()
    print("STATUS:")
    print(response["status"])

    print()
    print("ANSWER:")
    print(response["answer"])

    print()
    print("CONFIDENCE:")
    print(response["answer_confidence"])

    print()
    print("TEMPORAL INPUT:")
    print(response["temporal_input"])

    print()
    print("TRACE:")
    print(response["trace"])

    if response["status"] != "success":
        raise RuntimeError(
            "Temporal Agent Service returned "
            "a non-success status."
        )

    if response["answer"] != "yes":
        raise RuntimeError(
            "Unexpected answer for Test question 0."
        )

    if response["answer_confidence"] is None:
        raise RuntimeError(
            "Temporal answer confidence is missing."
        )

    if response["temporal_input"][
        "question_id"
    ] != 0:
        raise RuntimeError(
            "Unexpected question ID."
        )

    if response["temporal_input"][
        "pair_id"
    ] != 7308:
        raise RuntimeError(
            "Unexpected temporal pair ID."
        )

    print()
    print("=" * 80)
    print(
        "TEMPORAL AGENT SERVICE: PASS"
    )
    print("=" * 80)