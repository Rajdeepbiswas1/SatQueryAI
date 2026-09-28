from pathlib import Path
import tempfile

import numpy as np
import tifffile


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
        and p.stem.upper().endswith(
            f"_{band.upper()}"
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

    if tuple(tensor.shape[0:2]) != (
        1,
        9,
    ):
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


def run_case(
    service,
    s1_path: Path,
    s2_path: Path,
    query: str,
    options,
    label: str,
):
    kwargs = {
        "query": query,
        "paths": [
            s1_path,
            s2_path,
        ],
        "patch_id": (
            f"cross_modal_{label.lower()}_smoke"
        ),
    }

    if options is not None:
        kwargs["options"] = options

    result = service.execute(
        **kwargs
    )

    print()
    print(f"--- {label} ---")
    print("status:", result["status"])
    print(
        "analysis.task:",
        result["analysis"].get("task"),
    )
    print(
        "selected tool:",
        result["execution_trace"].get(
            "selected_tool"
        ),
    )
    print(
        "S1 feature shape:",
        result["feature_extraction"]["s1"]["shape"],
    )
    print(
        "S2 feature shape:",
        result["feature_extraction"]["s2"]["shape"],
    )
    print(
        "response:",
        result["response"],
    )

    assert result["status"] == "success"

    assert (
        result["analysis"]["task"]
        == "cross_modal"
    )

    assert (
        result["feature_extraction"]["s1"]["shape"]
        == [1, 196, 768]
    )

    assert (
        result["feature_extraction"]["s2"]["shape"]
        == [1, 196, 768]
    )

    return result


def main() -> None:
    from backend.cross_modal_upload_service import (
        SatQueryAICrossModalUploadService,
    )

    print(
        "Creating SatQueryAICrossModalUploadService..."
    )

    service = (
        SatQueryAICrossModalUploadService(
            device="cpu"
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="satqueryai_cross_modal_smoke_"
    ) as temp_dir:

        temp_root = Path(
            temp_dir
        )

        s1_path = (
            temp_root
            / "cross_s1_test.tif"
        )

        s2_path = (
            temp_root
            / "cross_s2_test.tif"
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

        # -------------------------------------------------------------
        # Binary cross-modal VQA
        # -------------------------------------------------------------

        binary_result = run_case(
            service=service,
            s1_path=s1_path,
            s2_path=s2_path,
            query=(
                "Is there water in the scene?"
            ),
            options=None,
            label="CROSS-MODAL BINARY",
        )

        # -------------------------------------------------------------
        # MCQ cross-modal VQA
        # -------------------------------------------------------------

        mcq_options = [
            "Water",
            "Built-up area",
            "Forest",
            "Agricultural land",
        ]

        mcq_result = run_case(
            service=service,
            s1_path=s1_path,
            s2_path=s2_path,
            query=(
                "Which type of land cover is most "
                "likely represented?"
            ),
            options=mcq_options,
            label="CROSS-MODAL MCQ",
        )

    print()
    print("=" * 80)
    print(
        "CROSS-MODAL UPLOAD SERVICE RESULTS"
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
        "CROSS-MODAL UPLOAD SERVICE: PASS"
    )


if __name__ == "__main__":
    main()