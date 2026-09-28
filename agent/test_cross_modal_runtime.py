from pathlib import Path
import torch

from agent.bootstrap import create_runtime_agent


PROJECT_ROOT = Path(r"D:\SatQueryAI")
DOFA_DIR = PROJECT_ROOT / "data" / "processed" / "dofa_features_download" / "dofa_features"


def main():
    print("=" * 80)
    print("SATQUERYAI — CROSS-MODAL END-TO-END SMOKE TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Find one real DOFA feature shard
    # ------------------------------------------------------------------
    shards = sorted(DOFA_DIR.glob("*.pt"))

    if not shards:
        raise FileNotFoundError(
            f"No .pt feature shards found in: {DOFA_DIR}"
        )

    shard_path = shards[0]

    print()
    print("Feature shard:")
    print(" ", shard_path)

    shard = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=False,
    )

    print()
    print("Shard keys:")
    print(" ", list(shard.keys()))

    # ------------------------------------------------------------------
    # 2. Extract first real S1/S2 pair
    # ------------------------------------------------------------------
    s1_all = shard["s1"]
    s2_all = shard["s2"]
    patch_ids = shard["patch_ids"]

    s1 = s1_all[0:1].float()
    s2 = s2_all[0:1].float()

    patch_id = str(patch_ids[0])

    print()
    print("Patch ID:")
    print(" ", patch_id)

    print("S1 shape:")
    print(" ", tuple(s1.shape))

    print("S2 shape:")
    print(" ", tuple(s2.shape))

    if s1.shape != (1, 196, 768):
        raise RuntimeError(f"Unexpected S1 shape: {tuple(s1.shape)}")

    if s2.shape != (1, 196, 768):
        raise RuntimeError(f"Unexpected S2 shape: {tuple(s2.shape)}")

    if not torch.isfinite(s1).all():
        raise RuntimeError("S1 contains non-finite values.")

    if not torch.isfinite(s2).all():
        raise RuntimeError("S2 contains non-finite values.")

    # ------------------------------------------------------------------
    # 3. Load the real binary question embedding cache
    # ------------------------------------------------------------------
    question_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "binary_question_embeddings"
    )

    q_files = sorted(question_dir.glob("*.pt"))

    if not q_files:
        raise FileNotFoundError(
            f"No question-embedding shards found in: {question_dir}"
        )

    q_path = q_files[0]

    print()
    print("Question shard:")
    print(" ", q_path)

    q_shard = torch.load(
        q_path,
        map_location="cpu",
        weights_only=False,
    )

    print()
    print("Question shard keys:")
    print(" ", list(q_shard.keys()))

    # Try the standard question embedding key.
    if "question_embeddings" not in q_shard:
        raise KeyError(
            "Expected 'question_embeddings' in question cache."
        )

    question_embeddings = q_shard["question_embeddings"]

    text_embedding = question_embeddings[0:1].float()

    print()
    print("Question embedding shape:")
    print(" ", tuple(text_embedding.shape))

    if text_embedding.shape != (1, 768):
        raise RuntimeError(
            f"Unexpected question embedding shape: "
            f"{tuple(text_embedding.shape)}"
        )

    if not torch.isfinite(text_embedding).all():
        raise RuntimeError(
            "Question embedding contains non-finite values."
        )

    # ------------------------------------------------------------------
    # 4. Build the real runtime agent
    # ------------------------------------------------------------------
    runtime_agent, grounding_tool, binary_vqa_tool = create_runtime_agent()

    print()
    print("Registered capabilities:")
    print(" ", runtime_agent.registry.capabilities())

    if not runtime_agent.registry.available("cross_modal"):
        raise RuntimeError(
            "Cross-modal capability is unavailable."
        )

    # ------------------------------------------------------------------
    # 5. Execute a real cross-modal query
    # ------------------------------------------------------------------
    query = (
        "Use the optical and SAR images together to identify "
        "whether the target region contains built-up or water-covered areas."
    )

    print()
    print("Query:")
    print(" ", query)

    result = runtime_agent.execute(
        query,
        s1=s1,
        s2=s2,
        text_embedding=text_embedding,
        patch_id=patch_id,
    )

    # ------------------------------------------------------------------
    # 6. Print the complete observable result
    # ------------------------------------------------------------------
    print()
    print("-" * 80)
    print("ANSWER")
    print("-" * 80)
    print(result.get("answer"))

    print()
    print("-" * 80)
    print("ANSWER STATUS")
    print("-" * 80)
    print(result.get("answer_status"))

    print()
    print("-" * 80)
    print("EVIDENCE")
    print("-" * 80)
    print(result.get("evidence"))

    print()
    print("-" * 80)
    print("EXECUTION TRACE")
    print("-" * 80)
    print(result.get("execution_trace"))

    print()
    print("=" * 80)
    print("CROSS-MODAL END-TO-END SMOKE TEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()