def _extract_evidence(
    output
):

    if not isinstance(output, dict):
        raise RuntimeError(
            "Specialist output must be a dictionary."
        )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Point specialist:
    #     heatmap
    #     pred_box
    #
    # C3 reference specialist:
    #     grounding_heatmap
    #     predicted_box
    # -------------------------------------------------------------------------

    heatmap = _find_named_tensor(
        output,
        [
            "heatmap",
            "grounding_heatmap",
            "heatmap_probs",
            "heatmap_logits",
            "grounding_logits",
            "match_heatmap",
        ]
    )

    if heatmap is None:
        raise RuntimeError(
            "No grounding heatmap found. "
            f"Available output keys: {list(output.keys())}"
        )

    probability_heatmap = _normalize_heatmap(
        heatmap
    )

    bbox = _find_named_tensor(
        output,
        [
            "pred_box",
            "predicted_box",
            "bbox",
            "box",
        ]
    )

    if bbox is not None:

        bbox = bbox.float()

        if bbox.ndim == 1:
            bbox = bbox.unsqueeze(0)

        if bbox.ndim != 2 or bbox.shape[-1] != 4:
            raise RuntimeError(
                f"Unexpected bbox shape: "
                f"{tuple(bbox.shape)}"
            )

        bbox = _sanitize_bbox(
            bbox
        )

        bbox_source = (
            "specialist_predicted_box"
        )

    else:

        bbox = _heatmap_moment_box_generic(
            probability_heatmap
        )

        bbox = _sanitize_bbox(
            bbox
        )

        bbox_source = (
            "heatmap_moments"
        )

    flat = probability_heatmap.flatten(
        1
    )

    peak_probability = flat.max(
        dim=1
    ).values

    spatial_entropy = -(
        probability_heatmap.clamp_min(
            1e-8
        )
        *
        probability_heatmap.clamp_min(
            1e-8
        ).log()
    ).sum(
        dim=(-2, -1)
    )

    return {
        "heatmap": probability_heatmap,
        "bbox": bbox,
        "bbox_source": bbox_source,
        "peak_probability": peak_probability,
        "spatial_entropy": spatial_entropy,
    }
