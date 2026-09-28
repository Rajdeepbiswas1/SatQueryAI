from pathlib import Path
import tempfile

import numpy as np
import tifffile
from fastapi import FastAPI
from fastapi.testclient import TestClient


S2_SCENE = Path(
    r"D:\SatQueryAI\data\processed\images_raw\BigEarthNet-S2"
    r"\S2B_MSIL2A_20180421T100029_N9999_R122_T33UXQ"
    r"\S2B_MSIL2A_20180421T100029_N9999_R122_T33UXQ_11_87"
)


REQUIRED_BANDS = (
    "B04",
    "B03",
    "B02",
    "B05",
    "B06",
    "B07",
    "B08",
    "B11",
    "B12",
)


def main() -> None:
    from backend.analyze_endpoint import router
    from backend.single_modality_upload_service import (
        SingleModalityUploadService,
    )

    # ---------------------------------------------------------------
    # Reconstruct the real S2 scene using the SAME application
    # upload adapter that was already verified successfully.
    # ---------------------------------------------------------------

    print("Reconstructing real BigEarthNet-S2 scene...")

    combined_tensor = (
        SingleModalityUploadService._load_s2_scene(
            S2_SCENE
        )
    )

    if combined_tensor.ndim != 4:
        raise RuntimeError(
            f"Unexpected reconstructed tensor shape: "
            f"{tuple(combined_tensor.shape)}"
        )

    if combined_tensor.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected batch dimension: "
            f"{tuple(combined_tensor.shape)}"
        )

    if combined_tensor.shape[1] != 9:
        raise RuntimeError(
            f"Expected 9 S2 channels, got "
            f"{combined_tensor.shape[1]}"
        )

    if not np.isfinite(
        combined_tensor.numpy()
    ).all():
        raise RuntimeError(
            "Reconstructed S2 tensor contains non-finite values."
        )

    print(
        "reconstructed_shape:",
        tuple(combined_tensor.shape),
    )

    # [1,C,H,W] -> [C,H,W]
    combined_array = (
        combined_tensor
        .squeeze(0)
        .numpy()
        .astype("float32")
    )

    # ---------------------------------------------------------------
    # Build a tiny temporary combined TIFF solely for the HTTP
    # multipart test. Production input contract remains unchanged.
    # ---------------------------------------------------------------

    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)

    with tempfile.TemporaryDirectory(
        prefix="satqueryai_single_s2_http_smoke_"
    ) as temp_dir:

        combined_path = (
            Path(temp_dir)
            / "single_s2_test.tif"
        )

        tifffile.imwrite(
            str(combined_path),
            combined_array,
        )

        with combined_path.open("rb") as file:

            response = client.post(
                "/analyze",
                data={
                    "query": "Is this primarily a built-up area?",
                    "compare_mode": "single",
                    "patch_id": "single_s2_http_smoke",
                },
                files=[
                    (
                        "image_files",
                        (
                            "single_s2_test.tif",
                            file,
                            "image/tiff",
                        ),
                    )
                ],
            )

    print("=" * 80)
    print("SATQUERYAI — SINGLE S2 /analyze HTTP SMOKE TEST")
    print("=" * 80)

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("Response:", response.text)
        raise RuntimeError(
            "Single S2 /analyze endpoint returned non-200."
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
        payload.get(
            "feature_extraction",
            {},
        ).get("shape"),
    )

    print(
        "task:",
        payload.get(
            "analysis",
            {},
        ).get("task"),
    )

    print(
        "selected_tool:",
        payload.get(
            "execution_trace",
            {},
        ).get("selected_tool"),
    )

    print(
        "routing_confidence:",
        payload.get(
            "execution_trace",
            {},
        ).get("routing_confidence"),
    )

    print(
        "response:",
        payload.get("response"),
    )

    assert payload["status"] == "success"

    assert (
        payload["input"]["modality"]
        == "s2"
    )

    assert (
        payload["feature_extraction"]["shape"]
        == [1, 196, 768]
    )

    assert (
        payload["analysis"]["task"]
        == "single_image_vqa"
    )

    assert (
        payload["execution_trace"]["selected_tool"]
        == "SingleModalityVQATool"
    )

    print()
    print(
        "SINGLE S2 /analyze HTTP ENDPOINT: PASS"
    )


if __name__ == "__main__":
    main()