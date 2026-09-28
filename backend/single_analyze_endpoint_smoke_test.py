from pathlib import Path
import tempfile

import tifffile
from fastapi import FastAPI
from fastapi.testclient import TestClient


S1_SCENE = Path(
    r"D:\SatQueryAI\data\processed\images_raw\BigEarthNet-S1"
    r"\S1A_IW_GRDH_1SDV_20171203T183450"
    r"\S1A_IW_GRDH_1SDV_20171203T183450_29SNB_17_89"
)


def find_band(scene: Path, band: str) -> Path:
    matches = [
        p for p in scene.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and p.stem.upper().endswith(f"_{band.upper()}")
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {band} TIFF, "
            f"found {len(matches)}."
        )

    return matches[0]


def main() -> None:
    from backend.analyze_endpoint import router

    vh_path = find_band(S1_SCENE, "VH")
    vv_path = find_band(S1_SCENE, "VV")

    vh = tifffile.imread(str(vh_path))
    vv = tifffile.imread(str(vv_path))

    if vh.ndim != 2 or vv.ndim != 2:
        raise RuntimeError(
            f"Expected 2D S1 bands, got "
            f"VH={vh.shape}, VV={vv.shape}"
        )

    if vh.shape != vv.shape:
        raise RuntimeError(
            f"VH/VV shape mismatch: "
            f"{vh.shape} vs {vv.shape}"
        )

    import numpy as np

    combined = np.stack(
        [
            vh,
            vv,
        ],
        axis=0,
    ).astype("float32")

    if combined.shape[0] != 2:
        raise RuntimeError(
            f"Unexpected combined S1 shape: {combined.shape}"
        )

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    with tempfile.TemporaryDirectory(
        prefix="satqueryai_single_http_smoke_"
    ) as temp_dir:

        combined_path = (
            Path(temp_dir)
            / "single_s1_test.tif"
        )

        tifffile.imwrite(
            str(combined_path),
            combined,
        )

        with combined_path.open("rb") as file:

            response = client.post(
                "/analyze",
                data={
                    "query": "Is there water in the image?",
                    "compare_mode": "single",
                    "patch_id": "single_s1_http_smoke",
                },
                files=[
                    (
                        "image_files",
                        (
                            "single_s1_test.tif",
                            file,
                            "image/tiff",
                        ),
                    )
                ],
            )

    print("=" * 80)
    print("SATQUERYAI — SINGLE /analyze HTTP SMOKE TEST")
    print("=" * 80)

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("Response:", response.text)
        raise RuntimeError(
            "Single-image /analyze endpoint returned non-200."
        )

    payload = response.json()

    print("status:", payload.get("status"))
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
        payload.get("execution_trace", {}).get("selected_tool"),
    )

    print(
        "routing_confidence:",
        payload.get("execution_trace", {}).get(
            "routing_confidence"
        ),
    )

    print(
        "response:",
        payload.get("response"),
    )

    assert payload["status"] == "success"
    assert payload["input"]["modality"] == "s1"
    assert payload["feature_extraction"]["shape"] == [
        1,
        196,
        768,
    ]
    assert payload["analysis"]["task"] == "single_image_vqa"
    assert (
        payload["execution_trace"]["selected_tool"]
        == "SingleModalityVQATool"
    )

    print()
    print("SINGLE /analyze HTTP ENDPOINT: PASS")


if __name__ == "__main__":
    main()