from pathlib import Path

from backend.single_modality_upload_service import SingleModalityUploadService


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


def check_result(label: str, result: dict, expected_modality: str):
    print(f"\n--- {label} ---")

    assert result["status"] == "success"
    assert result["input"]["modality"] == expected_modality

    feature_shape = result["feature_extraction"]["shape"]
    assert feature_shape == [1, 196, 768], feature_shape

    analysis = result["analysis"]
    assert analysis is not None
    assert analysis.get("task") == "single_image_vqa"

    print("status:", result["status"])
    print("modality:", result["input"]["modality"])
    print("input_kind:", result["input"]["input_kind"])
    print("feature_shape:", feature_shape)
    print("task:", analysis.get("task"))
    print("selected_tool:", analysis.get("selected_specialist"))
    print("response:", result["response"])


def main():
    print("Creating SingleModalityUploadService...")
    service = SingleModalityUploadService(device="cpu")

    print("\nTesting S1...")
    s1_result = service.execute(
        query="What type of scene is shown in this image?",
        path=S1_SCENE,
        modality="s1",
    )
    check_result("S1", s1_result, "s1")

    print("\nTesting S2...")
    s2_result = service.execute(
        query="What type of land cover is shown in this image?",
        path=S2_SCENE,
        modality="s2",
    )
    check_result("S2", s2_result, "s2")

    print("\nSINGLE-MODALITY UPLOAD SERVICE: PASS")


if __name__ == "__main__":
    main()