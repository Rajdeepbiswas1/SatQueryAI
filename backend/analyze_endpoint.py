from pathlib import Path
from typing import Any

import shutil
import tempfile
import base64

from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Form,
)

from backend.agent_inference_service import (
    AgentInferenceError,
    SatQueryAIAgentInferenceService,
)

from backend.temporal_inference_service import (
    SatQueryAITemporalInferenceService,
)

from backend.single_modality_upload_service import (
    SingleModalityUploadService,
)

from backend.single_modality_grounding_service import (
    SatQueryAISingleModalityGroundingInferenceService,
)

from backend.cross_modal_upload_service import (
    CrossModalUploadInferenceError,
    SatQueryAICrossModalUploadService,
)

from agent.core import SatQueryAIQueryAnalyzer
from agent.response_formatter import format_agent_response

from backend.upload_service import (
    UploadPreprocessingError,
    build_upload_channel_maps,
)

from backend.evidence_renderer import (
    EvidenceRenderingError,
    render_grounding_evidence,
)


router = APIRouter()


_inference_service = None
_temporal_inference_service = None
_single_modality_upload_service = None
_single_modality_grounding_service = None
_query_analyzer = None
_cross_modal_upload_service = None


def get_inference_service() -> SatQueryAIAgentInferenceService:
    global _inference_service

    if _inference_service is None:
        _inference_service = SatQueryAIAgentInferenceService(
            device="cpu"
        )

    return _inference_service


def get_temporal_inference_service() -> SatQueryAITemporalInferenceService:
    global _temporal_inference_service

    if _temporal_inference_service is None:
        _temporal_inference_service = SatQueryAITemporalInferenceService(
            device="cpu"
        )

    return _temporal_inference_service

def get_single_modality_upload_service() -> SingleModalityUploadService:
    global _single_modality_upload_service

    if _single_modality_upload_service is None:
        _single_modality_upload_service = SingleModalityUploadService(
            device="cpu"
        )

    return _single_modality_upload_service

def get_single_modality_grounding_service() -> SatQueryAISingleModalityGroundingInferenceService:
    global _single_modality_grounding_service

    if _single_modality_grounding_service is None:
        _single_modality_grounding_service = (
            SatQueryAISingleModalityGroundingInferenceService(
                device="cpu"
            )
        )

    return _single_modality_grounding_service


def get_query_analyzer() -> SatQueryAIQueryAnalyzer:
    global _query_analyzer

    if _query_analyzer is None:
        _query_analyzer = SatQueryAIQueryAnalyzer()

    return _query_analyzer

def _extract_grounding_bbox(
    agent_result: dict[str, Any],
) -> list[float] | None:
    """
    Extract a normalized grounding bounding box from an agent result.

    Supported evidence layouts:

        evidence["bbox"]

    or:

        evidence["data"]["bbox"]

    Tensor-like and numpy-like values are converted to Python lists.
    """

    evidence = agent_result.get(
        "evidence"
    )

    bbox_values = None

    if isinstance(
        evidence,
        dict,
    ):

        bbox_values = evidence.get(
            "bbox"
        )

        if bbox_values is None:

            evidence_data = evidence.get(
                "data"
            )

            if isinstance(
                evidence_data,
                dict,
            ):

                bbox_values = evidence_data.get(
                    "bbox"
                )

    if bbox_values is None:
        return None

    if hasattr(
        bbox_values,
        "detach",
    ):

        bbox_values = (
            bbox_values
            .detach()
            .cpu()
            .tolist()
        )

    elif hasattr(
        bbox_values,
        "tolist",
    ):

        bbox_values = (
            bbox_values
            .tolist()
        )

    if not isinstance(
        bbox_values,
        (list, tuple),
    ):
        return None

    if len(bbox_values) == 0:
        return None

    if (
        len(bbox_values) == 1
        and isinstance(
            bbox_values[0],
            (list, tuple),
        )
    ):
        bbox = bbox_values[0]

    else:
        bbox = bbox_values

    if isinstance(
        bbox,
        tuple,
    ):
        bbox = list(
            bbox
        )

    if (
        not isinstance(
            bbox,
            list,
        )
        or len(bbox) != 4
    ):
        return None

    try:
        return [
            float(value)
            for value in bbox
        ]

    except Exception:
        return None


def _render_grounding_visual_evidence(
    agent_result: dict[str, Any],
    source_path: Path,
    modality: str,
    temp_root: Path,
    query: str,
) -> dict[str, Any]:
    """
    Render grounding visual evidence for a combined application TIFF.

    This is an inference-time visualization step only. It does not
    modify the grounding model or its predicted bounding box.
    """

    analysis = agent_result.get(
        "analysis",
        {},
    )

    if not (
        isinstance(
            analysis,
            dict,
        )
        and analysis.get("task") == "grounding"
    ):
        return {
            "available": False,
        }

    bbox = _extract_grounding_bbox(
        agent_result
    )

    if bbox is None:
        return {
            "available": False,
            "reason": (
                "Grounding model did not return a valid "
                "bounding box."
            ),
        }

    evidence_png = (
        temp_root
        / "grounding_evidence.png"
    )

    try:

        rendered = render_grounding_evidence(
            s2_files=source_path,
            bbox=bbox,
            output_path=evidence_png,
            query=query,
            modality=modality,
        )

        png_bytes = evidence_png.read_bytes()

        encoded_png = (
            base64.b64encode(
                png_bytes
            )
            .decode(
                "ascii"
            )
        )

        return {
            "available": True,
            "format": "PNG",
            "mime_type": "image/png",
            "data_url": (
                "data:image/png;base64,"
                + encoded_png
            ),
            "image_width": (
                rendered[
                    "image_width"
                ]
            ),
            "image_height": (
                rendered[
                    "image_height"
                ]
            ),
            "bbox_normalized": (
                rendered[
                    "bbox_normalized"
                ]
            ),
            "bbox_pixels": (
                rendered[
                    "bbox_pixels"
                ]
            ),
            "rgb_bands": (
                rendered[
                    "rgb_bands"
                ]
            ),
        }

    except EvidenceRenderingError as exc:

        return {
            "available": False,
            "error": str(exc),
        }


def get_cross_modal_upload_service() -> SatQueryAICrossModalUploadService:
    global _cross_modal_upload_service

    if _cross_modal_upload_service is None:
        _cross_modal_upload_service = (
            SatQueryAICrossModalUploadService(
                device="cpu"
            )
        )

    return _cross_modal_upload_service

@router.post("/analyze")
async def analyze_real(
    query: str = Form(...),
    image_files: list[UploadFile] = File(...),
    patch_id: str = Form("uploaded_pair"),
    option_a: str | None = Form(None),
    option_b: str | None = Form(None),
    option_c: str | None = Form(None),
    option_d: str | None = Form(None),
    compare_mode: str = Form("optical-sar"),
) -> dict[str, Any]:

    query = query.strip()
    patch_id = patch_id.strip()
    compare_mode = compare_mode.strip().lower()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    if not patch_id:
        raise HTTPException(
            status_code=400,
            detail="patch_id cannot be empty.",
        )

    option_values = [
        option_a,
        option_b,
        option_c,
        option_d,
    ]

    any_options = any(
        value is not None
        for value in option_values
    )

    if any_options:

        if not all(
            value is not None
            and value.strip()
            for value in option_values
        ):
            raise HTTPException(
                status_code=400,
                detail="MCQ requests require all four options.",
            )

        options = [
            value.strip()
            for value in option_values
        ]

    else:
        options = None
        
    # ===============================================================
    # SINGLE-MODALITY APPLICATION-FACING PATH
    #
    # One combined TIFF containing:
    #   - 2 channels for S1, or
    #   - 9 trained channels for S2
    #
    # The single-modality upload service performs modality detection,
    # frozen DOFA extraction, text embedding, and agent dispatch.
    # ===============================================================

    if compare_mode == "single":

        if len(image_files) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Single-image analysis requires exactly "
                    "1 TIFF/GeoTIFF file."
                ),
            )

        upload = image_files[0]

        original_name = Path(
            upload.filename or ""
        ).name

        if not original_name:
            raise HTTPException(
                status_code=400,
                detail="Uploaded single-image file has no filename.",
            )

        suffix = Path(
            original_name
        ).suffix.lower()

        if suffix not in {
            ".tif",
            ".tiff",
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported single-image file type: "
                    f"{original_name}. "
                    "Only TIFF/GeoTIFF files are supported "
                    "for general single-image analysis."
                ),
            )

        try:

            with tempfile.TemporaryDirectory(
                prefix="satqueryai_single_upload_"
            ) as temp_dir:

                temp_root = Path(temp_dir)

                destination = (
                    temp_root
                    / original_name
                )

                with destination.open(
                    "wb"
                ) as output_file:

                    shutil.copyfileobj(
                        upload.file,
                        output_file,
                        length=1024 * 1024,
                    )

                if destination.stat().st_size == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Uploaded file is empty: "
                            f"{original_name}"
                        ),
                    )

                try:

                    query_analysis = (
                        get_query_analyzer().analyze(
                            query=query
                        )
                    )

                    if query_analysis.task == "grounding":

                        from backend.single_modality_upload_service import (
                            SingleModalityUploadService,
                        )

                        visual_tensor, resolved_modality = (
                            SingleModalityUploadService._read_combined_tiff(
                                destination,
                                None,
                            )
                        )

                        grounding_service = (
                            get_single_modality_grounding_service()
                        )

                        grounding_result = (
                            grounding_service.execute(
                                query=query,
                                image=visual_tensor,
                                modality=resolved_modality,
                                reference=query,
                            )
                        )

                        raw_agent_result = (
                            grounding_result["response"]
                        )

                        formatted_response = (
                            format_agent_response(
                                raw_agent_result
                            )
                        )

                        visual_evidence = (
                            _render_grounding_visual_evidence(
                                agent_result=raw_agent_result,
                                source_path=destination,
                                modality=resolved_modality,
                                temp_root=temp_root,
                                query=query,
                            )
                        )

                        return {
                            "status": "success",
                            "query": query,
                            "input": grounding_result["input"],
                            "feature_extraction": (
                                grounding_result[
                                    "feature_extraction"
                                ]
                            ),
                            "text_embedding": (
                                grounding_result[
                                    "text_embedding"
                                ]
                            ),
                            "analysis": (
                                grounding_result[
                                    "analysis"
                                ]
                            ),
                            "execution_trace": (
                                grounding_result[
                                    "execution_trace"
                                ]
                            ),
                            "response": formatted_response,
                            "visual_evidence": visual_evidence,
                        }

                    # ---------------------------------------------------------------
                    # Non-grounding single-image request → existing VQA path
                    # ---------------------------------------------------------------

                    service = (
                        get_single_modality_upload_service()
                    )

                    result = (
                        service.execute(
                            query=query,
                            path=destination,
                            options=options,
                        )
                    )

                except ValueError as exc:

                    raise HTTPException(
                        status_code=400,
                        detail=str(exc),
                    ) from exc

                except Exception as exc:

                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Single-modality inference failed: "
                            f"{exc}"
                        ),
                    ) from exc

                agent_result = result[
                    "agent_result"
                ]

                return {
                    "status": "success",
                    "query": result["query"],
                    "input": result["input"],
                    "feature_extraction": (
                        result["feature_extraction"]
                    ),
                    "text_embedding": (
                        result["text_embedding"]
                    ),
                    "option_embedding": (
                        result["option_embedding"]
                    ),
                    "analysis": (
                        agent_result["analysis"]
                    ),
                    "execution_trace": (
                        agent_result["execution_trace"]
                    ),
                    "response": result["response"],
                    "visual_evidence": {
                        "available": False,
                        "reason": (
                            "Single-image VQA currently returns "
                            "structured model evidence rather than "
                            "a rendered spatial overlay."
                        ),
                    },
                }

        except HTTPException:
            raise

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    "SatQueryAI single-image upload processing failed: "
                    f"{exc}"
                ),
            ) from exc

    # ===============================================================
    # TEMPORAL APPLICATION-FACING PATH
    #
    # Two spatially corresponding before/after images.
    # Supported general formats:
    #   TIFF / GeoTIFF
    # Benchmark formats:
    #   PNG / JPEG where permitted by the benchmark.
    #
    # This path deliberately does NOT enter the legacy 14-TIFF
    # Sentinel-1/Sentinel-2 paired preprocessing pipeline.
    # ===============================================================

    if compare_mode == "temporal":

        if len(image_files) != 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Temporal analysis requires exactly 2 images: "
                    "one BEFORE image and one AFTER image."
                ),
            )

        temporal_extensions = {
            ".tif",
            ".tiff",
            ".png",
            ".jpg",
            ".jpeg",
        }

        try:

            with tempfile.TemporaryDirectory(
                prefix="satqueryai_temporal_upload_"
            ) as temp_dir:

                temp_root = Path(temp_dir)

                saved_paths: list[Path] = []

                for index, upload in enumerate(
                    image_files
                ):

                    original_name = Path(
                        upload.filename or ""
                    ).name

                    if not original_name:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Uploaded temporal file "
                                f"#{index + 1} has no filename."
                            ),
                        )

                    suffix = Path(
                        original_name
                    ).suffix.lower()

                    if suffix not in temporal_extensions:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Unsupported temporal file type: "
                                f"{original_name}. "
                                "Supported formats are TIFF/GeoTIFF "
                                "and PNG/JPEG for permitted benchmark "
                                "inputs."
                            ),
                        )

                    destination = (
                        temp_root
                        / f"{index:02d}_{original_name}"
                    )

                    with destination.open(
                        "wb"
                    ) as output_file:

                        shutil.copyfileobj(
                            upload.file,
                            output_file,
                            length=1024 * 1024,
                        )

                    if destination.stat().st_size == 0:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Uploaded temporal file is empty: "
                                f"{original_name}"
                            ),
                        )

                    saved_paths.append(
                        destination
                    )

                try:

                    temporal_service = (
                        get_temporal_inference_service()
                    )

                    result = (
                        temporal_service.execute(
                            query=query,
                            paths=saved_paths,
                        )
                    )

                except ValueError as exc:

                    raise HTTPException(
                        status_code=400,
                        detail=str(exc),
                    ) from exc

                except Exception as exc:

                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Temporal inference failed: "
                            f"{exc}"
                        ),
                    ) from exc

                agent_result = result[
                    "agent_result"
                ]

                return {
                    "status": "success",
                    "query": result["query"],
                    "input": result["input"],
                    "feature_extraction": (
                        result["feature_extraction"]
                    ),
                    "text_embedding": (
                        result["text_embedding"]
                    ),
                    "analysis": (
                        agent_result["analysis"]
                    ),
                    "execution_trace": (
                        agent_result["execution_trace"]
                    ),
                    "response": result["response"],
                    "visual_evidence": {
                        "available": False,
                        "reason": (
                            "Temporal endpoint does not "
                            "render a grounding overlay."
                        ),
                    },
                }
                
        except HTTPException:
            raise

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    f"SatQueryAI temporal upload processing failed: "
                    f"{exc}"
                ),
            ) from exc

    # ===============================================================
    # OPTICAL + SAR APPLICATION-FACING PATH
    #
    # Two uploaded combined TIFFs:
    #     one S1
    #     one S2
    #
    # The new service performs modality detection and cross-modal
    # feature extraction.
    #
    # ===============================================================

    if compare_mode == "optical-sar" and len(image_files) == 2:

        try:

            with tempfile.TemporaryDirectory(
                prefix="satqueryai_cross_modal_upload_"
            ) as temp_dir:

                temp_root = Path(temp_dir)

                saved_paths: list[Path] = []

                for index, upload in enumerate(
                    image_files
                ):

                    original_name = Path(
                        upload.filename or ""
                    ).name

                    if not original_name:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Uploaded cross-modal file "
                                f"#{index + 1} has no filename."
                            ),
                        )

                    suffix = Path(
                        original_name
                    ).suffix.lower()

                    if suffix not in {
                        ".tif",
                        ".tiff",
                    }:

                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Unsupported cross-modal file type: "
                                f"{original_name}. "
                                "Only TIFF/GeoTIFF files are supported."
                            ),
                        )

                    destination = (
                        temp_root
                        / f"{index:02d}_{original_name}"
                    )

                    with destination.open(
                        "wb"
                    ) as output_file:

                        shutil.copyfileobj(
                            upload.file,
                            output_file,
                            length=1024 * 1024,
                        )

                    if destination.stat().st_size == 0:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Uploaded cross-modal file is empty: "
                                f"{original_name}"
                            ),
                        )

                    saved_paths.append(
                        destination
                    )

                try:

                    service = (
                        get_cross_modal_upload_service()
                    )

                    result = (
                        service.execute(
                            query=query,
                            paths=saved_paths,
                            options=options,
                            patch_id=patch_id,
                        )
                    )

                    visual_evidence = (
                        _render_grounding_visual_evidence(
                            agent_result=result["agent_result"],
                            source_path=Path(
                                result["input"]["s2"]["path"]
                            ),
                            modality="s2",
                            temp_root=temp_root,
                            query=query,
                        )
                    )

                except CrossModalUploadInferenceError as exc:

                    raise HTTPException(
                        status_code=400,
                        detail=str(exc),
                    ) from exc

                return {
                    "status": "success",
                    "query": result["query"],
                    "input": result["input"],
                    "feature_extraction": (
                        result["feature_extraction"]
                    ),
                    "text_embedding": (
                        result["text_embedding"]
                    ),
                    "option_embedding": (
                        result["option_embedding"]
                    ),
                    "analysis": result["analysis"],
                    "execution_trace": (
                        result["execution_trace"]
                    ),
                    "response": result["response"],
                    "visual_evidence": visual_evidence,
                }

        except HTTPException:
            raise

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    "SatQueryAI cross-modal upload processing failed: "
                    f"{exc}"
                ),
            ) from exc


    # ===============================================================
    # LEGACY / EXISTING PAIRED S1 + S2 PATH
    #
    # This branch intentionally preserves the existing 14-TIFF
    # production contract.
    # ===============================================================

    if len(image_files) != 14:
        raise HTTPException(
            status_code=400,
            detail=(
                "Current SatQueryAI paired-input endpoint requires "
                "exactly 14 TIFF files: 2 Sentinel-1 + "
                "12 Sentinel-2 channels."
            ),
        )

    try:

        with tempfile.TemporaryDirectory(
            prefix="satqueryai_upload_"
        ) as temp_dir:

            temp_root = Path(temp_dir)

            saved_paths: list[Path] = []

            for index, upload in enumerate(
                image_files
            ):

                original_name = Path(
                    upload.filename or ""
                ).name

                if not original_name:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Uploaded file #{index + 1} "
                            "has no filename."
                        ),
                    )

                suffix = Path(
                    original_name
                ).suffix.lower()

                if suffix not in {
                    ".tif",
                    ".tiff",
                }:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Unsupported file type: "
                            f"{original_name}. "
                            "Only TIFF/GeoTIFF files are "
                            "currently supported."
                        ),
                    )

                destination = (
                    temp_root
                    / f"{index:02d}_{original_name}"
                )

                with destination.open(
                    "wb"
                ) as output_file:

                    shutil.copyfileobj(
                        upload.file,
                        output_file,
                        length=1024 * 1024,
                    )

                if destination.stat().st_size == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Uploaded file is empty: "
                            f"{original_name}"
                        ),
                    )

                saved_paths.append(
                    destination
                )

            try:

                service = get_inference_service()

                result = service.execute(
                    query=query,
                    paths=saved_paths,
                    patch_id=patch_id,
                    options=options,
                )

            except UploadPreprocessingError as exc:

                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

            except AgentInferenceError as exc:

                raise HTTPException(
                    status_code=500,
                    detail=str(exc),
                ) from exc

            except ValueError as exc:

                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

            agent_result = result[
                "agent_result"
            ]

            # -------------------------------------------------------
            # Grounding visual evidence
            # -------------------------------------------------------

            visual_evidence = {
                "available": False,
            }

            analysis = agent_result.get(
                "analysis",
                {},
            )

            if (
                isinstance(
                    analysis,
                    dict,
                )
                and analysis.get("task")
                == "grounding"
            ):

                evidence = agent_result.get(
                    "evidence"
                )

                bbox_values = None

                if isinstance(
                    evidence,
                    dict,
                ):

                    bbox_values = evidence.get(
                        "bbox"
                    )

                    if bbox_values is None:

                        evidence_data = evidence.get(
                            "data"
                        )

                        if isinstance(
                            evidence_data,
                            dict,
                        ):

                            bbox_values = (
                                evidence_data.get(
                                    "bbox"
                                )
                            )

                if bbox_values is not None:

                    if hasattr(
                        bbox_values,
                        "detach",
                    ):
                        bbox_values = (
                            bbox_values
                            .detach()
                            .cpu()
                            .tolist()
                        )

                    elif hasattr(
                        bbox_values,
                        "tolist",
                    ):
                        bbox_values = (
                            bbox_values
                            .tolist()
                        )

                    if not isinstance(
                        bbox_values,
                        (list, tuple),
                    ):
                        bbox_values = None

                if (
                    isinstance(
                        bbox_values,
                        (list, tuple),
                    )
                    and len(bbox_values) > 0
                ):

                    if (
                        len(bbox_values) == 1
                        and isinstance(
                            bbox_values[0],
                            (list, tuple),
                        )
                    ):
                        bbox = bbox_values[0]

                    else:
                        bbox = bbox_values

                    if (
                        isinstance(
                            bbox,
                            tuple,
                        )
                    ):
                        bbox = list(
                            bbox
                        )

                    if (
                        not isinstance(
                            bbox,
                            list,
                        )
                        or len(bbox) != 4
                    ):
                        bbox = None

                else:

                    bbox = None

                if bbox is not None:

                    try:

                        _, s2_files = (
                            build_upload_channel_maps(
                                saved_paths
                            )
                        )

                        evidence_png = (
                            temp_root
                            / "grounding_evidence.png"
                        )

                        rendered = (
                            render_grounding_evidence(
                                s2_files=s2_files,
                                bbox=bbox,
                                output_path=evidence_png,
                                query=query,
                            )
                        )

                        png_bytes = (
                            evidence_png.read_bytes()
                        )

                        encoded_png = (
                            base64.b64encode(
                                png_bytes
                            )
                            .decode(
                                "ascii"
                            )
                        )

                        visual_evidence = {
                            "available": True,
                            "format": "PNG",
                            "mime_type": "image/png",
                            "data_url": (
                                "data:image/png;base64,"
                                + encoded_png
                            ),
                            "image_width": (
                                rendered[
                                    "image_width"
                                ]
                            ),
                            "image_height": (
                                rendered[
                                    "image_height"
                                ]
                            ),
                            "bbox_normalized": (
                                rendered[
                                    "bbox_normalized"
                                ]
                            ),
                            "bbox_pixels": (
                                rendered[
                                    "bbox_pixels"
                                ]
                            ),
                            "rgb_bands": (
                                rendered[
                                    "rgb_bands"
                                ]
                            ),
                        }

                    except EvidenceRenderingError as exc:

                        visual_evidence = {
                            "available": False,
                            "error": str(exc),
                        }

            return {
                "status": "success",
                "query": result["query"],
                "input": result["input"],
                "feature_extraction": (
                    result["feature_extraction"]
                ),
                "text_embedding": (
                    result["text_embedding"]
                ),
                "analysis": (
                    agent_result["analysis"]
                ),
                "execution_trace": (
                    agent_result["execution_trace"]
                ),
                "response": result["response"],
                "visual_evidence": visual_evidence,
            }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"SatQueryAI upload processing failed: "
                f"{exc}"
            ),
        ) from exc