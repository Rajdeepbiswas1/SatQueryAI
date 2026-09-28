from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.analyze_endpoint import router


T1 = Path(
    r"D:\SatQueryAI\data\processed\temporal_raw\im1\00003.png"
)

T2 = Path(
    r"D:\SatQueryAI\data\processed\temporal_raw\im2\00003.png"
)

QUERY = "Did the regions of non-vegetated ground surface change?"


def main() -> None:

    print("===================================================")
    print("SATQUERYAI TEMPORAL /analyze ENDPOINT SMOKE TEST")
    print("===================================================")

    assert T1.exists(), f"Missing T1: {T1}"
    assert T2.exists(), f"Missing T2: {T2}"

    print("Input files: PASS")

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    with (
        T1.open("rb") as before_file,
        T2.open("rb") as after_file,
    ):

        response = client.post(
            "/analyze",
            data={
                "query": QUERY,
                "patch_id": "temporal_smoke",
                "compare_mode": "temporal",
            },
            files=[
                (
                    "image_files",
                    (
                        "before.png",
                        before_file,
                        "image/png",
                    ),
                ),
                (
                    "image_files",
                    (
                        "after.png",
                        after_file,
                        "image/png",
                    ),
                ),
            ],
        )

    print("HTTP status:", response.status_code)

    assert response.status_code == 200, (
        f"Endpoint failed: {response.text}"
    )

    result = response.json()

    assert result["status"] == "success"

    analysis = result["analysis"]
    trace = result["execution_trace"]
    response_block = result["response"]

    print("Endpoint status: PASS")
    print("Detected task:", analysis.get("task"))
    print("Selected tool:", trace.get("selected_tool"))
    print("Response:", response_block)

    feature_extraction = result["feature_extraction"]

    print(
        "T1 shape:",
        feature_extraction.get("t1_shape"),
    )

    print(
        "T2 shape:",
        feature_extraction.get("t2_shape"),
    )

    print(
        "Diff shape:",
        feature_extraction.get("diff_shape"),
    )

    print(
        "TEMPORAL /analyze ENDPOINT: PASS"
    )


if __name__ == "__main__":
    main()