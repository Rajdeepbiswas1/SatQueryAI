from agent.bootstrap import create_runtime_agent
from agent.temporal_runtime_store import TemporalRuntimeStore
from agent.response_formatter import format_agent_response


def main():
    print("=" * 80)
    print("SATQUERYAI — TEMPORAL AGENT END-TO-END TEST")
    print("=" * 80)

    store = TemporalRuntimeStore()

    sample = store.get(
        question_id=0
    )

    runtime_agent = create_runtime_agent()[0]

    result = runtime_agent.execute(
        query=sample["question"],
        t1=sample["t1"],
        t2=sample["t2"],
        diff=sample["diff"],
        text_embedding=sample["text_embedding"],
    )

    response = format_agent_response(
        result
    )

    print()
    print("QUESTION:")
    print(sample["question"])

    print()
    print("EXPECTED:")
    print(sample["answer"])

    print()
    print("STATUS:")
    print(response["status"])

    print()
    print("ANSWER:")
    print(response["answer"])

    print()
    print("ANSWER CONFIDENCE:")
    print(response["answer_confidence"])

    print()
    print("ANALYSIS:")
    print(response["analysis"])

    print()
    print("RESULT:")
    print(response["result"])

    print()
    print("TRACE:")
    print(response["trace"])

    if response["status"] != "success":
        raise RuntimeError(
            "Temporal Agent execution failed."
        )

    if response["answer"] != sample["answer"]:
        raise RuntimeError(
            "Temporal Agent answer does not match "
            "the verified Test answer."
        )

    if response["analysis"]["task"] != (
        "temporal_change"
    ):
        raise RuntimeError(
            "Agent did not detect temporal_change."
        )

    print()
    print("=" * 80)
    print("TEMPORAL AGENT END-TO-END: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()