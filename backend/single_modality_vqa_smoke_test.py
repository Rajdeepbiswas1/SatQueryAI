from pathlib import Path

import torch

from backend.single_modality_vqa_service import (
    SatQueryAISingleModalityVQAInferenceService,
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


def main() -> None:

    print("===================================================")
    print("SATQUERYAI SINGLE-MODALITY VQA SMOKE TEST")
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
                "Could not find 14-channel tensor in source file."
            )

    else:
        raise RuntimeError(
            f"Unsupported source payload type: {type(payload)}"
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
    print("S1 raw shape:", list(s1_image.shape))
    print("S2 raw shape:", list(s2_image.shape))

    service = (
        SatQueryAISingleModalityVQAInferenceService(
            device="cpu"
        )
    )

    print("Inference service: INITIALIZED")

    tests = [
        {
            "name": "S1 Binary",
            "modality": "s1",
            "image": s1_image,
            "query": "Is there vegetation visible in this image?",
            "options": None,
        },
        {
            "name": "S2 Binary",
            "modality": "s2",
            "image": s2_image,
            "query": "Is there vegetation visible in this image?",
            "options": None,
        },
        {
            "name": "S1 MCQ",
            "modality": "s1",
            "image": s1_image,
            "query": "Which type of land cover is most likely visible?",
            "options": [
                "Vegetation",
                "Water",
                "Bare ground",
                "Snow or ice",
            ],
        },
        {
            "name": "S2 MCQ",
            "modality": "s2",
            "image": s2_image,
            "query": "Which type of land cover is most likely visible?",
            "options": [
                "Vegetation",
                "Water",
                "Bare ground",
                "Snow or ice",
            ],
        },
    ]

    results = []

    for test in tests:

        print("")
        print("---------------------------------------------------")
        print(test["name"])
        print("---------------------------------------------------")

        result = service.execute(
            query=test["query"],
            image=test["image"],
            modality=test["modality"],
            options=test["options"],
        )

        results.append(result)

        assert result["status"] == "success"

        assert result["analysis"]["task"] == (
            "single_image_vqa"
        )

        assert result["input"]["modality"] == (
            test["modality"]
        )

        assert result["input"]["feature_shape"] == [
            1,
            196,
            768,
        ]

        trace = result["execution_trace"]

        print(
            "Detected task:",
            result["analysis"]["task"],
        )

        print(
            "Routing confidence:",
            result["analysis"].get(
                "routing_confidence"
            ),
        )

        print(
            "Selected specialist:",
            trace.get(
                "selected_tool"
            )
            or trace.get(
                "selected_specialist"
            ),
        )

        print(
            "Response:",
            result["response"].get(
                "evidence",
                result["response"],
            ),
        )

        print("STATUS: PASS")

    # -----------------------------------------------------------
    # Explicit routing checks
    # -----------------------------------------------------------

    specialists = []

    for result in results:

        evidence = result["response"].get(
            "evidence",
            {}
        )

        if isinstance(evidence, dict):

            data = evidence.get(
                "data",
                {}
            )

            if isinstance(data, dict):

                specialist = data.get(
                    "specialist"
                )

                if specialist:
                    specialists.append(
                        specialist
                    )

        trace = result["response"].get(
            "execution_trace",
            {}
        )

        if isinstance(trace, dict):

            specialist = (
                trace.get(
                    "selected_specialist"
                )
            )

            if specialist:
                specialists.append(
                    specialist
                )

    print("")
    print("===================================================")
    print("SINGLE-MODALITY VQA EXECUTION COMPLETE")
    print("===================================================")

    for result in results:
        print(
            result["input"]["modality"],
            "|",
            "MCQ" if result["response"].get(
                "evidence",
                {},
            ).get(
                "data",
                {},
            ).get(
                "question_type"
            ) else "Binary",
            "|",
            result["status"],
        )

    print("")
    print(
        "SINGLE-MODALITY VQA INTEGRATION: PASS"
    )


if __name__ == "__main__":
    main()