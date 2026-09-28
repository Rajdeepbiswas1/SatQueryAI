from pathlib import Path

import torch

from backend.single_modality_grounding_service import (
    SatQueryAISingleModalityGroundingInferenceService,
)


SOURCE = Path(
    r"D:\SatQueryAI\data\processed\preprocessed"
    r"\S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_40_90.pt"
)

S1_INDICES = [0, 1]

S2_INDICES = [
    5,
    4,
    3,
    6,
    7,
    8,
    9,
    11,
    12,
]


def find_key_recursive(
    value,
    target_key: str,
):
    if isinstance(value, dict):

        if target_key in value:
            return value[target_key]

        for nested in value.values():

            found = find_key_recursive(
                nested,
                target_key,
            )

            if found is not None:
                return found

    elif isinstance(value, (list, tuple)):

        for nested in value:

            found = find_key_recursive(
                nested,
                target_key,
            )

            if found is not None:
                return found

    return None


def normalize_bbox(value):

    if value is None:
        return None

    if isinstance(value, torch.Tensor):

        value = (
            value
            .detach()
            .cpu()
            .tolist()
        )

    elif hasattr(value, "tolist"):

        value = value.tolist()

    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(
            value[0],
            (list, tuple),
        )
    ):
        value = value[0]

    if not isinstance(
        value,
        (list, tuple),
    ):
        return None

    if len(value) != 4:
        return None

    try:
        bbox = [
            float(x)
            for x in value
        ]
    except Exception:
        return None

    return bbox


def main() -> None:

    print("===================================================")
    print("SATQUERYAI SINGLE-MODALITY GROUNDING SMOKE TEST")
    print("===================================================")

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing source sample: {SOURCE}"
        )

    payload = torch.load(
        SOURCE,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(payload, torch.Tensor):
        image = payload

    elif isinstance(payload, dict):

        image = None

        for value in payload.values():

            if isinstance(value, torch.Tensor):

                if tuple(value.shape) in {
                    (14, 120, 120),
                    (1, 14, 120, 120),
                }:
                    image = value
                    break

        if image is None:
            raise RuntimeError(
                "Could not find 14-channel tensor."
            )

    else:
        raise RuntimeError(
            f"Unsupported payload type: {type(payload)}"
        )

    if image.ndim == 4:
        image = image[0]

    if tuple(image.shape) != (
        14,
        120,
        120,
    ):
        raise RuntimeError(
            f"Unexpected source shape: {tuple(image.shape)}"
        )

    image = image.float()

    s1_image = image[
        S1_INDICES
    ].unsqueeze(0)

    s2_image = image[
        S2_INDICES
    ].unsqueeze(0)

    print("Source:", SOURCE)
    print(
        "S1 raw shape:",
        list(s1_image.shape),
    )
    print(
        "S2 raw shape:",
        list(s2_image.shape),
    )

    service = (
        SatQueryAISingleModalityGroundingInferenceService(
            device="cpu"
        )
    )

    print("Inference service: INITIALIZED")

    tests = [
        (
            "S1",
            s1_image,
            "s1",
        ),
        (
            "S2",
            s2_image,
            "s2",
        ),
    ]

    for name, image_tensor, modality in tests:

        print("")
        print("---------------------------------------------------")
        print(name, "REFERENCE GROUNDING")
        print("---------------------------------------------------")

        result = service.execute(
            query="Where is the largest urban area in the image?",
            image=image_tensor,
            modality=modality,
            reference="largest urban area",
        )

        assert result["status"] == "success"

        assert result["analysis"]["task"] == (
            "grounding"
        )

        assert result["analysis"]["grounding_mode"] == (
            "reference"
        )

        assert result["input"]["modality"] == (
            modality
        )

        assert result["input"]["feature_shape"] == [
            1,
            196,
            768,
        ]

        response = result["response"]

        assert isinstance(
            response,
            dict,
        )

        # The outer agent handler is expected to be the
        # application-level grounding dispatcher.
        outer_specialist = find_key_recursive(
            result,
            "selected_tool",
        )

        inner_specialist = find_key_recursive(
            response,
            "selected_specialist",
        )

        if inner_specialist is None:
            inner_specialist = find_key_recursive(
                response,
                "specialist",
            )

        if inner_specialist is None:

            evidence = response.get(
                "evidence",
                {}
            )

            if isinstance(
                evidence,
                dict,
            ):

                evidence_data = evidence.get(
                    "data",
                    {}
                )

                if isinstance(
                    evidence_data,
                    dict,
                ):
                    inner_specialist = (
                        evidence_data.get(
                            "specialist"
                        )
                    )

                if inner_specialist is None:

                    execution_trace = (
                        evidence.get(
                            "execution_trace",
                            {}
                        )
                    )

                    if isinstance(
                        execution_trace,
                        dict,
                    ):
                        inner_specialist = (
                            execution_trace.get(
                                "selected_specialist"
                            )
                        )

        bbox_value = find_key_recursive(
            response,
            "bbox",
        )

        if bbox_value is None:
            bbox_value = find_key_recursive(
                response,
                "predicted_box",
            )

        bbox = normalize_bbox(
            bbox_value
        )

        print(
            "Detected task:",
            result["analysis"]["task"],
        )

        print(
            "Grounding mode:",
            result["analysis"]["grounding_mode"],
        )

        print(
            "Outer tool:",
            outer_specialist,
        )

        print(
            "Inner specialist:",
            inner_specialist,
        )

        print(
            "BBox:",
            bbox,
        )

        assert outer_specialist in {
            "GroundingModalityDispatcher",
            "UnifiedGroundingRouter",
        }

        assert inner_specialist is not None

        assert (
            "SingleModality"
            in str(inner_specialist)
        )

        assert bbox is not None

        assert len(bbox) == 4

        assert all(
            0.0 <= x <= 1.0
            for x in bbox
        )

        print("Spatial evidence: PASS")
        print("STATUS: PASS")

    print("")
    print("===================================================")
    print(
        "SINGLE-MODALITY GROUNDING INTEGRATION: PASS"
    )
    print("===================================================")


if __name__ == "__main__":
    main()