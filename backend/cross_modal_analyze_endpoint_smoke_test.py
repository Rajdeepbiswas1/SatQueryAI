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
        if (
            p.is_file()
            and p.suffix.lower() in {".tif", ".tiff"}
            and p.stem.upper().endswith(
                f"_{band.upper()}"
            )
        )
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

    array = np.stack(
        [vh, vv],
        axis=0,
    ).astype("float32")

    tifffile.imwrite(
        str(path),
        array,
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

    if tuple(tensor.shape[:2]) != (
        1,
        9,
    ):
        raise RuntimeError(
            f"Unexpected S2 reconstruction shape: "
            f"{tuple(tensor.shape)}"
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


def post_request(
    client: TestClient,
    image_paths: list[Path],
    query: str,
    options: list[str] | None,
    label: str,
):
    data = {
        "query": query,
        "compare_mode": "optical-sar",
        "patch_id": f"cross_modal_http_{label.lower()}_smoke",
    }

    if options is not None:
        data.update(
            {
                "option_a": options[0],
                "option_b": options[1],
                "option_c": options[2],
                "option_d": options[3],
            }
        )

    opened_files = []

    try:
        files = []

        for path in image_paths:
            file_handle = path.open("rb")
            opened_files.append(file_handle)

            files.append(
                (
                    "image_files",
                    (
                        path.name,
                        file_handle,
                        "image/tiff",
                    ),
                )
            )

        response = client.post(
            "/analyze",
            data=data,
            files=files,
        )

    finally:
        for file_handle in opened_files:
            file_handle.close()

    print()
    print(f"--- {label} ---")
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
            f"{label} /analyze request failed."
        )

    payload = response.json()

    print(
        "status:",
        payload.get("status"),
    )

    print(
        "S1 input:",
        payload.get("input", {}).get("s1", {}).get(
            "input_kind"
        ),
    )

    print(
        "S2 input:",
        payload.get("input", {}).get("s2", {}).get(
            "input_kind"
        ),
    )

    print(
        "S1 feature shape:",
        payload.get(
            "feature_extraction",
            {},
        ).get(
            "s1",
            {},
        ).get(
            "shape"
        ),
    )

    print(
        "S2 feature shape:",
        payload.get(
            "feature_extraction",
            {},
        ).get(
            "s2",
            {},
        ).get(
            "shape"
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
        payload.get("response"),
    )

    assert payload["status"] == "success"

    assert (
        payload["analysis"]["task"]
        == "cross_modal"
    )

    assert (
        payload["execution_trace"]["selected_tool"]
        == "CrossModalVQARouter"
    )

    assert (
        payload["feature_extraction"]["s1"]["shape"]
        == [1, 196, 768]
    )

    assert (
        payload["feature_extraction"]["s2"]["shape"]
        == [1, 196, 768]
    )

    return payload


def main() -> None:
    from backend.analyze_endpoint import router

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    with tempfile.TemporaryDirectory(
        prefix="satqueryai_cross_modal_http_smoke_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        s1_path = (
            temp_root
            / "cross_modal_s1_test.tif"
        )

        s2_path = (
            temp_root
            / "cross_modal_s2_test.tif"
        )

        print(
            "Building temporary combined S1 TIFF..."
        )

        build_s1(
            s1_path
        )

        print(
            "Building temporary combined S2 TIFF..."
        )

        build_s2(
            s2_path
        )

        binary_result = post_request(
            client=client,
            image_paths=[
                s1_path,
                s2_path,
            ],
            query="Is there water in the scene?",
            options=None,
            label="CROSS-MODAL BINARY",
        )

        mcq_result = post_request(
            client=client,
            image_paths=[
                s1_path,
                s2_path,
            ],
            query=(
                "Which type of land cover is most "
                "likely represented?"
            ),
            options=[
                "Water",
                "Built-up area",
                "Forest",
                "Agricultural land",
            ],
            label="CROSS-MODAL MCQ",
        )

    print()
    print("=" * 80)
    print(
        "CROSS-MODAL /analyze HTTP RESULTS"
    )
    print("=" * 80)

    print(
        "Binary task:",
        binary_result["analysis"]["task"],
    )

    print(
        "MCQ task:",
        mcq_result["analysis"]["task"],
    )

    print()
    print(
        "CROSS-MODAL /analyze HTTP ENDPOINT: PASS"
    )


if __name__ == "__main__":
    main()