from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import tifffile
import torch
import torch.nn.functional as F

from agent.bootstrap import create_runtime_agent
from agent.response_formatter import format_agent_response
from backend.single_modality_feature_service import SingleModalityFeatureService
from backend.text_embedding_service import SatQueryAITextEmbeddingService


class SingleModalityUploadService:
    """
    Application-facing service for single-modality remote-sensing input.

    Supported application layouts:
      1. A combined TIFF containing exactly 2 channels (S1)
      2. A combined TIFF containing exactly 9 channels (S2 trained-band stack)
      3. An S1 scene directory containing VH.tif + VV.tif
      4. An S2 scene directory containing the BigEarthNet 12-band TIFFs.
         The exact 9 trained bands are selected automatically.

    Frozen model contracts:
      S1 -> [B, 2, H, W]
      S2 -> [B, 9, H, W]
      Feature representation -> [B, 196, 768]
    """

    S1_REQUIRED_FILES = {
        "vh": "VH",
        "vv": "VV",
    }

    # Exact nine S2 bands used by the existing frozen DOFA extractor.
    S2_REQUIRED_BANDS = (
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

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        self.feature_service = SingleModalityFeatureService(
            device=device
        )
        self.text_embedding_service = SatQueryAITextEmbeddingService(
            device=device
        )

        # create_runtime_agent() returns:
        #   (agent, grounding_tool, binary_vqa_tool)
        self.agent, _, _ = create_runtime_agent()

    # ------------------------------------------------------------------
    # TIFF loading
    # ------------------------------------------------------------------

    @staticmethod
    def _read_single_tiff(path: Path) -> torch.Tensor:
        """Read one TIFF band as a finite float32 [H, W] tensor."""
        if not path.is_file():
            raise ValueError(f"TIFF file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in {".tif", ".tiff"}:
            raise ValueError(f"Unsupported input file type: {path.suffix}")

        array = tifffile.imread(str(path))

        if array.ndim != 2:
            # Permit a singleton channel dimension only.
            if array.ndim == 3 and 1 in array.shape:
                array = array.squeeze()
            else:
                raise ValueError(
                    f"Expected a single 2D TIFF band at {path}, "
                    f"got shape {array.shape}"
                )

        tensor = torch.as_tensor(array, dtype=torch.float32)

        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite values found in TIFF: {path}")

        return tensor

    @classmethod
    def _read_combined_tiff(
        cls,
        path: Path,
        modality: Optional[str],
    ) -> tuple[torch.Tensor, str]:
        """
        Read a combined multi-channel TIFF.

        Returns:
            tensor [1, C, H, W]
            resolved modality ('s1' or 's2')
        """
        array = tifffile.imread(str(path))

        if array.ndim == 2:
            raise ValueError(
                "A single-band TIFF is insufficient for this single-modality "
                "pipeline. S1 requires VH+VV and S2 requires the trained "
                "9-band stack."
            )

        if array.ndim != 3:
            raise ValueError(
                f"Expected a 3D multi-channel TIFF, got shape {array.shape}"
            )

        # Detect CHW vs HWC by the known trained channel counts.
        if array.shape[0] in {2, 9}:
            tensor = torch.as_tensor(array, dtype=torch.float32)
        elif array.shape[-1] in {2, 9}:
            tensor = torch.as_tensor(array, dtype=torch.float32).permute(
                2, 0, 1
            )
        else:
            raise ValueError(
                "Combined TIFF must contain exactly 2 channels (S1) or "
                "9 channels (S2 trained-band stack). "
                f"Received shape {array.shape}."
            )

        channels = tensor.shape[0]

        detected_modality = "s1" if channels == 2 else "s2"

        if modality is not None and modality.lower() != detected_modality:
            raise ValueError(
                f"Requested modality '{modality}' does not match the "
                f"detected {detected_modality} input."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite values found in combined TIFF.")

        return tensor.unsqueeze(0), detected_modality

    # ------------------------------------------------------------------
    # Scene-directory handling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_name(path: Path) -> str:
        return path.name.upper()

    @classmethod
    def _find_s1_band(
        cls,
        files: Sequence[Path],
        band_name: str,
    ) -> Optional[Path]:
        band_name = band_name.upper()

        for path in files:
            name = cls._normalize_name(path)
            stem = path.stem.upper()

            if (
                name.endswith(f"_{band_name}.TIF")
                or name.endswith(f"_{band_name}.TIFF")
                or stem.endswith(f"_{band_name}")
                or stem == band_name
            ):
                return path

        return None

    @classmethod
    def _find_s2_band(
        cls,
        files: Sequence[Path],
        band_name: str,
    ) -> Optional[Path]:
        target = f"_{band_name.upper()}"

        for path in files:
            stem = path.stem.upper()

            if stem.endswith(target):
                return path

        return None

    @classmethod
    def _load_s1_scene(
        cls,
        scene_dir: Path,
    ) -> torch.Tensor:
        """
        Load VH + VV in the fixed two-channel order:

            channel 0 = VH
            channel 1 = VV
        """
        files = [
            p for p in scene_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        ]

        vh_path = cls._find_s1_band(files, "VH")
        vv_path = cls._find_s1_band(files, "VV")

        if vh_path is None or vv_path is None:
            raise ValueError(
                "S1 scene must contain both VH.tif and VV.tif. "
                f"Scene: {scene_dir}"
            )

        vh = cls._read_single_tiff(vh_path)
        vv = cls._read_single_tiff(vv_path)

        if vh.shape != vv.shape:
            raise ValueError(
                "S1 VH and VV bands have different shapes: "
                f"VH={tuple(vh.shape)}, VV={tuple(vv.shape)}"
            )

        return torch.stack([vh, vv], dim=0).unsqueeze(0)

    @classmethod
    def _load_s2_scene(
        cls,
        scene_dir: Path,
    ) -> torch.Tensor:
        """
        Load the nine S2 bands used by the frozen DOFA extractor.

        Order is intentionally fixed to:

            B04, B03, B02, B05, B06,
            B07, B08, B11, B12
        """
        files = [
            p for p in scene_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        ]

        band_tensors: list[torch.Tensor] = []
        missing: list[str] = []

        for band_name in cls.S2_REQUIRED_BANDS:
            band_path = cls._find_s2_band(files, band_name)

            if band_path is None:
                missing.append(band_name)
                continue

            band_tensors.append(
                cls._read_single_tiff(band_path)
            )

        if missing:
            raise ValueError(
                "S2 scene is missing required trained bands: "
                + ", ".join(missing)
            )

        # BigEarthNet band layouts can contain differing native spatial
        # resolutions. Use B04 as the reference grid because it is the
        # first band in the frozen trained ordering.
        target_h, target_w = band_tensors[0].shape

        normalized_bands: list[torch.Tensor] = []

        for band in band_tensors:
            if band.shape != (target_h, target_w):
                band = F.interpolate(
                    band.unsqueeze(0).unsqueeze(0),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            normalized_bands.append(band)

        tensor = torch.stack(normalized_bands, dim=0)

        if not torch.isfinite(tensor).all():
            raise ValueError("Non-finite values found in S2 scene.")

        return tensor.unsqueeze(0)

    @classmethod
    def _load_scene(
        cls,
        path: Path,
        modality: Optional[str],
    ) -> tuple[torch.Tensor, str, dict]:
        """
        Resolve the supplied path.

        A directory is treated as a complete BigEarthNet scene.
        A combined TIFF is treated as a directly usable channel stack.
        """
        requested = modality.lower() if modality else None

        if path.is_dir():
            files = [
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
            ]

            names = [p.name.upper() for p in files]

            has_vh = any(name.endswith("_VH.TIF") or name.endswith("_VH.TIFF") for name in names)
            has_vv = any(name.endswith("_VV.TIF") or name.endswith("_VV.TIFF") for name in names)

            s2_presence = {
                band: any(
                    p.stem.upper().endswith(f"_{band}")
                    for p in files
                )
                for band in cls.S2_REQUIRED_BANDS
            }

            has_s2 = all(s2_presence.values())

            if requested == "s1":
                tensor = cls._load_s1_scene(path)
                resolved = "s1"
            elif requested == "s2":
                tensor = cls._load_s2_scene(path)
                resolved = "s2"
            elif has_vh and has_vv:
                tensor = cls._load_s1_scene(path)
                resolved = "s1"
            elif has_s2:
                tensor = cls._load_s2_scene(path)
                resolved = "s2"
            else:
                raise ValueError(
                    "Could not infer a supported single-modality scene. "
                    "Expected either an S1 directory with VH+VV or an "
                    "S2 directory containing the nine required trained bands."
                )

            metadata = {
                "input_kind": "scene_directory",
                "scene_directory": str(path),
                "selected_files": [
                    str(p) for p in files
                ],
            }

            return tensor, resolved, metadata

        if path.is_file():
            tensor, resolved = cls._read_combined_tiff(
                path,
                requested,
            )

            metadata = {
                "input_kind": "combined_tiff",
                "source_file": str(path),
            }

            return tensor, resolved, metadata

        raise ValueError(f"Input path does not exist: {path}")

    # ------------------------------------------------------------------
    # Request validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_options(
        options: Optional[Sequence[str]],
    ) -> Optional[list[str]]:
        if options is None:
            return None

        values = [str(x).strip() for x in options]

        if len(values) != 4:
            raise ValueError(
                "MCQ requests require exactly four options (A-D)."
            )

        if any(not value for value in values):
            raise ValueError("MCQ options cannot be empty.")

        return values

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def execute(
        self,
        query: str,
        path: str | Path,
        modality: Optional[str] = None,
        options: Optional[Sequence[str]] = None,
    ) -> dict:
        query = str(query).strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        input_path = Path(path)

        if modality is not None:
            modality = modality.lower().strip()

            if modality not in {"s1", "s2"}:
                raise ValueError(
                    "Modality must be 's1' or 's2'."
                )

        validated_options = self._validate_options(options)

        # --------------------------------------------------------------
        # Question eligibility preflight
        # --------------------------------------------------------------
        #
        # Validate the query before reading the image or running the
        # frozen feature/text encoders. MCQ requests remain allowed
        # because the presence of four validated options identifies
        # the dedicated MCQ pathway.

        query_analysis = self.agent.analyze_query(
            query=query,
        )

        self.agent.validate_question_eligibility(
            analysis=query_analysis,
            query=query,
            options=validated_options,
        )

        # --------------------------------------------------------------
        # Input reconstruction
        # --------------------------------------------------------------
        
        visual, resolved_modality, input_metadata = self._load_scene(
            input_path,
            modality,
        )

        visual = visual.to(self.device)

        if resolved_modality == "s1" and visual.shape[1] != 2:
            raise RuntimeError(
                f"S1 reconstruction produced {visual.shape[1]} channels; "
                "expected 2."
            )

        if resolved_modality == "s2" and visual.shape[1] != 9:
            raise RuntimeError(
                f"S2 reconstruction produced {visual.shape[1]} channels; "
                "expected 9."
            )

        # --------------------------------------------------------------
        # Frozen DOFA feature extraction
        # --------------------------------------------------------------

        features, feature_metadata = self.feature_service.extract(
            visual,
            modality=resolved_modality,
        )

        if tuple(features.shape) != (1, 196, 768):
            raise RuntimeError(
                "Unexpected single-modality feature shape: "
                f"{tuple(features.shape)}; expected (1, 196, 768)."
            )

        # --------------------------------------------------------------
        # Text embedding
        # --------------------------------------------------------------

        question_embedding, question_metadata = (
            self.text_embedding_service.embed(query)
        )

        # Normalize service output to [1, 768].
        if question_embedding.ndim == 1:
            question_embedding = question_embedding.unsqueeze(0)

        question_embedding = question_embedding.to(self.device)

        if tuple(question_embedding.shape) != (1, 768):
            raise RuntimeError(
                "Unexpected question embedding shape: "
                f"{tuple(question_embedding.shape)}; expected (1, 768)."
            )

        # --------------------------------------------------------------
        # MCQ option embeddings
        # --------------------------------------------------------------

        option_embedding = None

        if validated_options is not None:
            option_vectors = []

            for option in validated_options:
                vector, _ = self.text_embedding_service.embed(option)

                if vector.ndim == 1:
                    vector = vector.unsqueeze(0)

                if tuple(vector.shape) != (1, 768):
                    raise RuntimeError(
                        "Unexpected option embedding shape for "
                        f"'{option}': {tuple(vector.shape)}"
                    )

                option_vectors.append(vector.to(self.device))

            option_embedding = torch.cat(
                option_vectors,
                dim=0,
            ).unsqueeze(0)

            if tuple(option_embedding.shape) != (1, 4, 768):
                raise RuntimeError(
                    "Unexpected option embedding shape: "
                    f"{tuple(option_embedding.shape)}; expected (1, 4, 768)."
                )

        # --------------------------------------------------------------
        # Agent execution
        # --------------------------------------------------------------

        agent_inputs = {
            "s1": features if resolved_modality == "s1" else None,
            "s2": features if resolved_modality == "s2" else None,
        }

        if option_embedding is not None:
            agent_inputs["options"] = option_embedding

        # Natural-language grounding uses the submitted query as its
        # textual reference when no separate reference is available.
        agent_inputs["reference"] = query

        agent_result = self.agent.execute(
            query,
            **agent_inputs,
            text_embedding=question_embedding,
        )
        
        formatted = format_agent_response(
            agent_result
        )

        task = (
            agent_result.get("analysis", {}).get("task")
            if isinstance(agent_result, dict)
            else None
        )

        if task != "single_image_vqa":
            raise RuntimeError(
                "Single-modality upload service expected the agent to "
                f"select 'single_image_vqa', but got {task!r}."
            )

        return {
            "status": "success",
            "query": query,
            "input": {
                "path": str(input_path),
                "modality": resolved_modality,
                **input_metadata,
            },
            "feature_extraction": {
                "service": "SingleModalityFeatureService",
                "modality": resolved_modality,
                "shape": list(features.shape),
            },
            "text_embedding": {
                "query_shape": list(question_embedding.shape),
            },
            "option_embedding": (
                {
                    "shape": list(option_embedding.shape),
                    "count": 4,
                }
                if option_embedding is not None
                else None
            ),
            "analysis": agent_result.get("analysis"),
            "execution_trace": agent_result.get("execution_trace"),
            "agent_result": agent_result,
            "response": formatted,
        }


__all__ = ["SingleModalityUploadService"]