from pathlib import Path
import re
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.bootstrap import create_runtime_agent
from agent.mcq_vqa_tool import SatQueryAIMCQVQATool


EXPECTED_PARAMS = 1_380_097
TEST_ID = 6479


def load_dofa_patch(patch_id: str):
    """
    Locate a patch in the verified DOFA shard cache.

    Cache contract:
        patch_ids
        s1
        s2
    """
    dofa_dir = (
        ROOT
        / "data"
        / "processed"
        / "dofa_features_download"
        / "dofa_features"
    )

    shards = sorted(dofa_dir.glob("shard_*.pt"))

    if not shards:
        raise FileNotFoundError(
            f"No DOFA shards found in {dofa_dir}"
        )

    for shard_path in shards:
        data = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        patch_ids = [
            str(x)
            for x in data["patch_ids"]
        ]

        for index, current_id in enumerate(patch_ids):
            if current_id == patch_id:
                s1 = data["s1"][index]
                s2 = data["s2"][index]

                return (
                    shard_path,
                    index,
                    s1,
                    s2,
                )

    raise RuntimeError(
        f"Patch not found in DOFA cache: {patch_id}"
    )


def load_mcq_option_embeddings(question_index: int):
    """
    Locate question + A/B/C/D embeddings in the option-aware cache.
    """
    option_dir = (
        ROOT
        / "data"
        / "processed"
        / "mcq_option_embeddings"
    )

    shards = sorted(
        option_dir.glob("shard_*.pt")
    )

    if not shards:
        raise FileNotFoundError(
            f"No MCQ option shards found in {option_dir}"
        )

    for shard_path in shards:
        data = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        indices = [
            int(x)
            for x in data["question_indices"].tolist()
        ]

        for index, current_index in enumerate(indices):
            if current_index != question_index:
                continue

            question = data[
                "question_embeddings"
            ][index]

            options = torch.stack(
                [
                    data[
                        "option_a_embeddings"
                    ][index],

                    data[
                        "option_b_embeddings"
                    ][index],

                    data[
                        "option_c_embeddings"
                    ][index],

                    data[
                        "option_d_embeddings"
                    ][index],
                ],
                dim=0,
            )

            return (
                shard_path,
                index,
                question,
                options,
            )

    raise RuntimeError(
        "Question index not found in MCQ option cache: "
        f"{question_index}"
    )


def parse_mcq_input(text: str):
    """
    Parse:
        question: a) ... b) ... c) ... d) ...

    Returns:
        question_text
        option_texts = [A, B, C, D]
    """
    pattern = re.compile(
        r"^(.*?)\s+"
        r"a\)\s*(.*?)\s+"
        r"b\)\s*(.*?)\s+"
        r"c\)\s*(.*?)\s+"
        r"d\)\s*(.*?)\s*$",
        flags=re.IGNORECASE,
    )

    match = pattern.match(
        text.strip()
    )

    if not match:
        raise RuntimeError(
            "Could not parse MCQ input into "
            "question + four A-D options.\n"
            f"Input:\n{text}"
        )

    question = match.group(1).strip()

    options = [
        match.group(2).strip(),
        match.group(3).strip(),
        match.group(4).strip(),
        match.group(5).strip(),
    ]

    return question, options


def main():
    print("=" * 80)
    print("SATQUERYAI REAL MCQ VQA AGENT SMOKE TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Build the verified runtime
    # ------------------------------------------------------------------
    runtime = create_runtime_agent()
    agent = runtime[0]

    print("\nRegistered capabilities before MCQ test:")
    print(agent.registry.capabilities())

    if "single_image_vqa" not in (
        agent.registry.capabilities()
    ):
        raise RuntimeError(
            "single_image_vqa capability is not registered."
        )

    print("\nAgent bootstrap: PASS")

    # ------------------------------------------------------------------
    # 2. Put the VERIFIED MCQ tool into this test-local agent
    # ------------------------------------------------------------------
    #
    # The normal bootstrap currently has Binary VQA registered under
    # single_image_vqa. The registry stores one handler per capability.
    #
    # Therefore this smoke test temporarily replaces that handler on
    # this in-memory agent instance only.
    #
    # bootstrap.py is NOT modified.
    # core.py is NOT modified.
    # Binary checkpoint is NOT modified.
    # MCQ checkpoint is NOT modified.
    #
    mcq_tool = SatQueryAIMCQVQATool()

    agent.registry.register(
        capability="single_image_vqa",
        tool_name="OptionAwareMCQVQA",
        handler=mcq_tool,
    )

    print(
        "\nTest-local VQA registration:",
        agent.registry.get("single_image_vqa")["tool_name"],
    )

    if (
        agent.registry.get(
            "single_image_vqa"
        )["tool_name"]
        != "OptionAwareMCQVQA"
    ):
        raise RuntimeError(
            "MCQ tool was not installed into the test-local registry."
        )

    print("MCQ test-local registration: PASS")

    # ------------------------------------------------------------------
    # 3. Load the real MCQ parquet
    # ------------------------------------------------------------------
    split_path = (
        ROOT
        / "data"
        / "processed"
        / "mcq_vqa_split.parquet"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing MCQ split: {split_path}"
        )

    df = pd.read_parquet(
        split_path
    )

    print(
        f"\nMCQ rows: {len(df):,}"
    )

    print("Columns:")
    print(list(df.columns))

    # Use the real, previously verified row.
    matches = df[
        df["ID"].astype(int) == TEST_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one row with ID={TEST_ID}, "
            f"found {len(matches)}."
        )

    row = matches.iloc[0]

    patch_id = str(
        row["patch_id"]
    )

    mcq_input = str(
        row["input"]
    )

    expected = str(
        row["output"]
    ).strip().upper()

    print("\nSelected MCQ row:")
    print("  ID:", int(row["ID"]))
    print("  Patch:", patch_id)
    print("  Type:", row["type"])
    print("  Category:", row["category"])
    print("  Expected:", expected)

    print("\nRaw MCQ input:")
    print(" ", mcq_input)

    # ------------------------------------------------------------------
    # 4. Parse question and A-D text
    # ------------------------------------------------------------------
    question_text, option_texts = (
        parse_mcq_input(mcq_input)
    )

    print("\nParsed question:")
    print(" ", question_text)

    print("\nParsed options:")
    for label, option in zip(
        ["A", "B", "C", "D"],
        option_texts,
    ):
        print(
            f"  {label}: {option}"
        )

    if len(option_texts) != 4:
        raise RuntimeError(
            "MCQ does not contain exactly four options."
        )

    print("MCQ parsing: PASS")

    # ------------------------------------------------------------------
    # 5. Load real DOFA features
    # ------------------------------------------------------------------
    (
        dofa_shard,
        dofa_row,
        s1,
        s2,
    ) = load_dofa_patch(
        patch_id
    )

    print("\nDOFA lookup:")
    print("  Shard:", dofa_shard)
    print("  Row:", dofa_row)
    print("  S1:", tuple(s1.shape))
    print("  S2:", tuple(s2.shape))

    # ------------------------------------------------------------------
    # 6. Load real question + option embeddings
    # ------------------------------------------------------------------
    (
        option_shard,
        option_row,
        question_embedding,
        options,
    ) = load_mcq_option_embeddings(
        TEST_ID
    )

    print("\nMCQ option-cache lookup:")
    print("  Shard:", option_shard)
    print("  Row:", option_row)
    print(
        "  Question:",
        tuple(question_embedding.shape),
    )
    print(
        "  Options:",
        tuple(options.shape),
    )

    # ------------------------------------------------------------------
    # 7. Prepare exact runtime tensors
    # ------------------------------------------------------------------
    s1 = torch.as_tensor(
        s1,
        dtype=torch.float32,
    ).unsqueeze(0)

    s2 = torch.as_tensor(
        s2,
        dtype=torch.float32,
    ).unsqueeze(0)

    question_embedding = torch.as_tensor(
        question_embedding,
        dtype=torch.float32,
    ).unsqueeze(0)

    options = torch.as_tensor(
        options,
        dtype=torch.float32,
    ).unsqueeze(0)

    print("\nRuntime tensor shapes:")
    print(
        "  S1:",
        tuple(s1.shape),
    )
    print(
        "  S2:",
        tuple(s2.shape),
    )
    print(
        "  Question:",
        tuple(question_embedding.shape),
    )
    print(
        "  Options:",
        tuple(options.shape),
    )

    assert tuple(s1.shape) == (
        1,
        196,
        768,
    )

    assert tuple(s2.shape) == (
        1,
        196,
        768,
    )

    assert tuple(question_embedding.shape) == (
        1,
        768,
    )

    assert tuple(options.shape) == (
        1,
        4,
        768,
    )

    print("Input contract: PASS")

    # ------------------------------------------------------------------
    # 8. Execute through the REAL agent
    # ------------------------------------------------------------------
    print("\nExecuting agent.execute()...")

    start = time.perf_counter()

    result = agent.execute(
        query=mcq_input,
        s1=s1,
        s2=s2,
        text_embedding=question_embedding,
        options=options,
        patch_id=patch_id,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    print(
        f"\nAgent execution time: "
        f"{elapsed_ms:.2f} ms"
    )

    # ------------------------------------------------------------------
    # 9. Print structured result
    # ------------------------------------------------------------------
    print("\nAgent result:")
    print(result)

    if result.get(
        "answer_status"
    ) != "success":
        raise RuntimeError(
            "Agent MCQ execution failed.\n"
            f"Result:\n{result}"
        )

    print("\nAgent status: SUCCESS")

    # ------------------------------------------------------------------
    # 10. Inspect routing trace
    # ------------------------------------------------------------------
    analysis = result.get(
        "analysis",
        {},
    )

    evidence = result.get(
        "evidence",
        {},
    )

    trace = result.get(
        "execution_trace",
        {},
    )

    print("\nAnalysis:")
    print(analysis)

    print("\nEvidence:")
    print(evidence)

    print("\nExecution trace:")
    print(trace)

    if analysis.get(
        "task"
    ) != "single_image_vqa":
        raise RuntimeError(
            "Agent did not classify the MCQ request "
            "as single_image_vqa."
        )

    print(
        "\nTask routing: PASS"
    )

    # ------------------------------------------------------------------
    # 11. Verify selected tool
    # ------------------------------------------------------------------
    selected_tool = trace.get(
        "selected_tool"
    )

    print(
        "Selected tool:",
        selected_tool,
    )

    if selected_tool != "OptionAwareMCQVQA":
        raise RuntimeError(
            "Agent selected the wrong VQA specialist.\n"
            f"Selected: {selected_tool}"
        )

    print(
        "MCQ specialist dispatch: PASS"
    )

    # ------------------------------------------------------------------
    # 12. Extract predicted answer
    # ------------------------------------------------------------------
    predicted = (
        evidence.get("answer")
        or evidence.get("predicted_answer")
        or evidence.get("prediction")
    )

    print(
        "\nPredicted answer:",
        predicted,
    )

    print(
        "Expected answer:",
        expected,
    )

    if predicted is not None:
        predicted = str(
            predicted
        ).strip().upper()

        if predicted == expected:
            print(
                "Prediction matches stored answer: PASS"
            )
        else:
            print(
                "Prediction differs from stored answer."
            )
            print(
                "Specialist execution still succeeded; "
                "this is not an integration failure."
            )

    # ------------------------------------------------------------------
    # 13. Final smoke-test result
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SATQUERYAI REAL MCQ VQA AGENT SMOKE TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()