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
            f"Expected exactly one {band} file, "
            f"found {len(matches)} in {scene}"
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
            f"S1 VH/VV shape mismatch: "
            f"{vh.shape} vs {vv.shape}"
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

    if tuple(tensor.shape[:2]) != (1, 9):
        raise RuntimeError(
            f"Unexpected S2 shape: {tuple(tensor.shape)}"
        )

    array = (
        tensor
        .squeeze(0)
        .numpy()
        .astype("float32")
    )

    tifffile.imwrite(
        str(path),
        array,
    )


def main() -> None:
    from backend.analyze_endpoint import router

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    with tempfile.TemporaryDirectory(
        prefix="satqueryai_cross_grounding_http_smoke_"
    ) as temp_dir:

        root = Path(temp_dir)

        s1_path = root / "cross_grounding_s1.tif"
        s2_path = root / "cross_grounding_s2.tif"

        print("Building S1 combined TIFF...")
        build_s1(s1_path)

        print("Building S2 combined TIFF...")
        build_s2(s2_path)

        query = (
            "Where is the largest urban area in the image?"
        )

        opened = []

        try:
            files = []

            for path in (
                s1_path,
                s2_path,
            ):
                handle = path.open("rb")
                opened.append(handle)

                files.append(
                    (
                        "image_files",
                        (
                            path.name,
                            handle,
                            "image/tiff",
                        ),
                    )
                )

            response = client.post(
                "/analyze",
                data={
                    "query": query,
                    "compare_mode": "optical-sar",
                    "patch_id": "cross_modal_grounding_smoke",
                },
                files=files,
            )

        finally:
            for handle in opened:
                handle.close()

    print("=" * 80)
    print(
        "SATQUERYAI — CROSS-MODAL GROUNDING /analyze SMOKE TEST"
    )
    print("=" * 80)

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
            "Cross-modal grounding endpoint returned non-200."
        )

    payload = response.json()

    print(
        "status:",
        payload.get("status"),
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
        payload.get("response"),
    )

    assert payload["status"] == "success"

    assert (
        payload["analysis"]["task"]
        == "grounding"
    )

    assert (
        payload["analysis"]["grounding_mode"]
        == "reference"
    )

    assert (
        payload["execution_trace"]["selected_tool"]
        == "CrossModalVQARouter"
        or
        payload["execution_trace"]["selected_tool"]
        == "GroundingModalityDispatcher"
    )

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

    if bbox is None:
        raise RuntimeError(
            "Cross-modal grounding response did not "
            "contain a bounding box."
        )

    print(
        "bbox:",
        bbox,
    )

    print()
    print(
        "CROSS-MODAL GROUNDING /analyze: PASS"
    )


if __name__ == "__main__":
    main()