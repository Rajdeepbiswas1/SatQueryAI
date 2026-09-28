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


def build_s1(path: Path) -> None:
    vh = tifffile.imread(
        str(find_band(S1_SCENE, "VH"))
    )
    vv = tifffile.imread(
        str(find_band(S1_SCENE, "VV"))
    )

    if vh.shape != vv.shape:
        raise RuntimeError(
            f"S1 VH/VV mismatch: {vh.shape} vs {vv.shape}"
        )

    combined = np.stack(
        [vh, vv],
        axis=0,
    ).astype("float32")

    tifffile.imwrite(
        str(path),
        combined,
    )


def build_s2(path: Path) -> None:
    from backend.single_modality_upload_service import (
        SingleModalityUploadService,
    )

    tensor = (
        SingleModalityUploadService._load_s2_scene(
            S2_SCENE
        )
    )

    if tuple(tensor.shape[0:2]) != (1, 9):
        raise RuntimeError(
            f"Unexpected S2 shape: {tuple(tensor.shape)}"
        )

    array = (
        tensor
        .squeeze(0)
        .numpy()
        .astype("float32")
    )

    if not np.isfinite(array).all():
        raise RuntimeError(
            "S2 tensor contains non-finite values."
        )

    tifffile.imwrite(
        str(path),
        array,
    )


def run_grounding(
    client: TestClient,
    image_path: Path,
    label: str,
):
    query = (
        "Where is the largest urban area in the image?"
    )

    with image_path.open("rb") as file:

        response = client.post(
            "/analyze",
            data={
                "query": query,
                "compare_mode": "single",
                "patch_id": (
                    f"single_grounding_{label.lower()}_smoke"
                ),
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
    print(f"--- {label} GROUNDING ---")
    print(
        "HTTP status:",
        response.status_code,
    )

    if response.status_code != 200:
        print(
            "Response:",
            response.text,
        )
        raise RuntimeError(
            f"{label} grounding endpoint failed."
        )

    payload = response.json()

    print(
        "status:",
        payload.get("status"),
    )

    print(
        "modality:",
        payload.get(
            "input",
            {},
        ).get(
            "modality"
        ),
    )

    print(
        "feature_shape:",
        payload.get(
            "feature_extraction",
            {},
        ).get(
            "feature_shape"
        ),
    )

    print(
        "task:",
        payload.get(
            "analysis",
            {},
        ).get(
            "task"
        ),
    )

    print(
        "grounding_mode:",
        payload.get(
            "analysis",
            {},
        ).get(
            "grounding_mode"
        ),
    )

    print(
        "selected_tool:",
        payload.get(
            "execution_trace",
            {},
        ).get(
            "selected_tool"
        ),
    )

    print(
        "response:",
        payload.get(
            "response"
        ),
    )

    assert payload["status"] == "success"

    assert payload["input"]["modality"] == label.lower()

    assert payload["analysis"]["task"] == "grounding"

    assert payload["execution_trace"]["selected_tool"] in {
        "GroundingModalityDispatcher",
        "SingleModalityGrounding",
        "SingleModalityGroundingV1",
    }

    response_data = payload["response"]

    assert isinstance(
        response_data,
        dict,
    )

    evidence = response_data.get(
        "evidence",
        {},
    )

    assert isinstance(
        evidence,
        dict,
    )

    evidence_data = evidence.get(
        "data",
        evidence,
    )

    assert isinstance(
        evidence_data,
        dict,
    )

    bbox = (
        evidence_data.get("bbox")
        or evidence_data.get("predicted_box")
    )

    assert bbox is not None

    print(
        "bbox:",
        bbox,
    )

    return payload


def main() -> None:
    from backend.analyze_endpoint import router

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    with tempfile.TemporaryDirectory(
        prefix="satqueryai_single_grounding_http_smoke_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        s1_path = (
            temp_root
            / "single_s1_grounding_test.tif"
        )

        s2_path = (
            temp_root
            / "single_s2_grounding_test.tif"
        )

        print(
            "Building temporary S1 combined TIFF..."
        )
        build_s1(s1_path)

        print(
            "Building temporary S2 combined TIFF..."
        )
        build_s2(s2_path)

        s1_result = run_grounding(
            client,
            s1_path,
            "s1",
        )

        s2_result = run_grounding(
            client,
            s2_path,
            "s2",
        )

    print()
    print("=" * 80)
    print(
        "SINGLE-MODALITY GROUNDING /analyze RESULTS"
    )
    print("=" * 80)

    print(
        "S1 task:",
        s1_result["analysis"]["task"],
    )

    print(
        "S2 task:",
        s2_result["analysis"]["task"],
    )

    print()
    print(
        "SINGLE-MODALITY GROUNDING /analyze: PASS"
    )


if __name__ == "__main__":
    main()