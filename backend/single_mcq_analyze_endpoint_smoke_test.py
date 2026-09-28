from pathlib import Path
import tempfile

import numpy as np
import tifffile
from fastapi import FastAPI
from fastapi.testclient import TestClient


S1_SCENE = Path(
    r"D:\SatQueryAI\data\processed\images_raw\BigEarthNet-S1"
    r"\S1A_IW_GRDH_1SDV_20171203T183450"
    r"\S1A_IW_GRDH_1SDV_20171203T183450_29SNB_17_89"
)

S2_SCENE = Path(
    r"D:\SatQueryAI\data\processed\images_raw\BigEarthNet-S2"
    r"\S2B_MSIL2A_20180421T100029_N9999_R122_T33UXQ"
    r"\S2B_MSIL2A_20180421T100029_N9999_R122_T33UXQ_11_87"
)


def find_band(scene: Path, band: str) -> Path:
    matches = [
        p
        for p in scene.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and p.stem.upper().endswith(f"_{band.upper()}")
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {band} file in {scene}, "
            f"found {len(matches)}."
        )

    return matches[0]


def build_s1_tiff(path: Path) -> None:
    vh = tifffile.imread(str(find_band(S1_SCENE, "VH")))
    vv = tifffile.imread(str(find_band(S1_SCENE, "VV")))

    if vh.shape != vv.shape:
        raise RuntimeError(
            f"S1 VH/VV shape mismatch: {vh.shape} vs {vv.shape}"
        )

    combined = np.stack(
        [vh, vv],
        axis=0,
    ).astype("float32")

    tifffile.imwrite(str(path), combined)


def build_s2_tiff(path: Path) -> None:
    from backend.single_modality_upload_service import (
        SingleModalityUploadService,
    )

    tensor = SingleModalityUploadService._load_s2_scene(
        S2_SCENE
    )

    if tuple(tensor.shape) != (
        1,
        9,
        tensor.shape[2],
        tensor.shape[3],
    ):
        raise RuntimeError(
            f"Unexpected S2 reconstruction shape: {tuple(tensor.shape)}"
        )

    array = (
        tensor
        .squeeze(0)
        .numpy()
        .astype("float32")
    )

    if not np.isfinite(array).all():
        raise RuntimeError(
            "S2 reconstructed tensor contains non-finite values."
        )

    tifffile.imwrite(
        str(path),
        array,
    )


def run_request(
    client: TestClient,
    image_path: Path,
    query: str,
    options: list[str],
    label: str,
):
    with image_path.open("rb") as file:

        response = client.post(
            "/analyze",
            data={
                "query": query,
                "compare_mode": "single",
                "patch_id": f"single_mcq_{label.lower()}_smoke",
                "option_a": options[0],
                "option_b": options[1],
                "option_c": options[2],
                "option_d": options[3],
            },
            files=[
                (
                    "image_files",
                    (
                        image_path.name,
                        file,
                        "image/tiff",
                    ),
                )
            ],
        )

    print()
    print(f"--- {label} MCQ ---")
    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("Response:", response.text)
        raise RuntimeError(
            f"{label} MCQ endpoint request failed."
        )

    payload = response.json()

    print(
        "status:",
        payload.get("status"),
    )

    print(
        "modality:",
        payload.get("input", {}).get("modality"),
    )

    print(
        "feature_shape:",
        payload.get("feature_extraction", {}).get("shape"),
    )

    print(
        "task:",
        payload.get("analysis", {}).get("task"),
    )

    print(
        "selected_tool:",
        payload.get("execution_trace", {}).get(
            "selected_tool"
        ),
    )

    print(
        "response:",
        payload.get("response"),
    )

    assert payload["status"] == "success"

    assert (
        payload["analysis"]["task"]
        == "single_image_vqa"
    )

    assert (
        payload["execution_trace"]["selected_tool"]
        == "SingleModalityVQATool"
    )

    assert (
        payload["feature_extraction"]["shape"]
        == [1, 196, 768]
    )

    response_data = payload["response"]

    assert isinstance(response_data, dict)

    evidence = response_data.get("evidence", {})

    if isinstance(evidence, dict):
        evidence_data = evidence.get("data", evidence)

        specialist = (
            evidence_data.get("specialist")
            if isinstance(evidence_data, dict)
            else None
        )

        print(
            "specialist:",
            specialist,
        )

        if specialist is not None:
            assert specialist in {
                "SingleModalityS1MCQVQA",
                "SingleModalityS2MCQVQA",
            }

    return payload


def main() -> None:
    from backend.analyze_endpoint import router

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    s1_options = [
        "Water",
        "Urban area",
        "Forest",
        "Agricultural land",
    ]

    s2_options = [
        "Water",
        "Built-up area",
        "Forest",
        "Agricultural land",
    ]

    with tempfile.TemporaryDirectory(
        prefix="satqueryai_single_mcq_http_smoke_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        s1_path = (
            temp_root
            / "single_s1_mcq_test.tif"
        )

        s2_path = (
            temp_root
            / "single_s2_mcq_test.tif"
        )

        print("Building temporary S1 combined TIFF...")
        build_s1_tiff(s1_path)

        print("Building temporary S2 combined TIFF...")
        build_s2_tiff(s2_path)

        s1_result = run_request(
            client=client,
            image_path=s1_path,
            query="Which type of area is most likely represented?",
            options=s1_options,
            label="S1",
        )

        s2_result = run_request(
            client=client,
            image_path=s2_path,
            query="Which type of land cover is most likely represented?",
            options=s2_options,
            label="S2",
        )

    print()
    print("=" * 80)
    print("SINGLE-MODALITY MCQ /analyze RESULTS")
    print("=" * 80)

    print(
        "S1 modality:",
        s1_result["input"]["modality"],
    )

    print(
        "S2 modality:",
        s2_result["input"]["modality"],
    )

    print()
    print("SINGLE-MODALITY MCQ /analyze: PASS")


if __name__ == "__main__":
    main()