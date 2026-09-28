# =============================================================================
# SATQUERYAI — AGENT CORE
# =============================================================================
#
# Reusable application-level orchestration layer.
#
# Responsibilities:
#   1. Natural-language query analysis
#   2. Capability selection
#   3. Capability registry lookup
#   4. Capability-specific input validation
#   5. Specialist execution
#   6. Structured evidence propagation
#   7. Observable execution trace
#
# Important:
#   - No model training occurs here.
#   - No checkpoint is modified here.
#   - Specialist implementations remain independent.
#   - Unavailable capabilities never fabricate results.
# =============================================================================

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional
import re
import time


# =============================================================================
# 1. AGENT DATA STRUCTURES
# =============================================================================

@dataclass
class QueryAnalysis:
    """
    Structured representation of the user's request.

    confidence is confidence in the routing decision, NOT confidence in
    specialist model correctness.
    """

    task: str
    grounding_mode: Optional[str]
    confidence: float
    matched_signals: list
    normalized_query: str


@dataclass
class AgentExecutionTrace:
    """
    Observable record of one agent execution.
    """

    query: str
    normalized_query: str
    detected_task: str
    grounding_mode: Optional[str]
    selected_tool: Optional[str]
    capability_available: bool
    status: str
    routing_confidence: float
    matched_signals: list
    validation_ms: float
    execution_ms: float
    total_ms: float


# =============================================================================
# 2. CAPABILITY REGISTRY
# =============================================================================

class SatQueryAICapabilityRegistry:
    """
    Registry of executable specialist capabilities.

    A specialist is registered using:
        capability name
        tool name
        callable handler
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        capability: str,
        tool_name: str,
        handler: Callable,
    ) -> None:

        capability = str(capability).strip().lower()
        tool_name = str(tool_name).strip()

        if not capability:
            raise ValueError(
                "Capability name cannot be empty."
            )

        if not tool_name:
            raise ValueError(
                "Tool name cannot be empty."
            )

        if not callable(handler):
            raise TypeError(
                "Tool handler must be callable."
            )

        self._tools[capability] = {
            "tool_name": tool_name,
            "handler": handler,
        }

    def unregister(
        self,
        capability: str,
    ) -> None:

        capability = str(capability).strip().lower()
        self._tools.pop(capability, None)

    def available(
        self,
        capability: str,
    ) -> bool:

        capability = str(capability).strip().lower()

        return capability in self._tools

    def get(
        self,
        capability: str,
    ) -> Optional[Dict[str, Any]]:

        capability = str(capability).strip().lower()

        return self._tools.get(capability)

    def capabilities(self) -> list:
        return sorted(self._tools.keys())


# =============================================================================
# 3. NATURAL-LANGUAGE QUERY ANALYZER
# =============================================================================

class SatQueryAIQueryAnalyzer:
    """
    Deterministic first-stage query analyzer.

    The current implementation deliberately uses observable linguistic
    signals so routing is reproducible and auditable.

    This component can later be replaced with a learned/LLM-based router
    without changing the agent execution contract.
    """

    TASK_SIGNAL_GROUPS = {

        "temporal_change": [
            "change",
            "changed",
            "changes",
            "before and after",
            "compare over time",
            "temporal",
            "time series",
            "previous image",
            "earlier image",
            "later image",
            "between 2020 and 2024",
            "between 2021 and 2024",
            "between 2022 and 2024",
            "growth",
            "decline",
            "increase",
            "decrease",
        ],

        "cross_modal": [
            "sar",
            "optical and sar",
            "optical vs sar",
            "optical versus sar",
            "compare optical and sar",
            "radar",
            "cross modal",
            "cross-modal",
            "multimodal",
            "multi modal",
        ],

        "single_image_vqa": [
            "how many",
            "what is",
            "what are",
            "which type",
            "what kind",
            "identify",
            "classify",
            "classification",
            "is there",
            "are there",
            "does the image contain",
            "describe the object",
            "tell me what",
        ],

        "captioning": [
            "caption",
            "generate a caption",
            "describe this image",
            "describe the image",
            "summarize the image",
            "image description",
        ],

        "grounding": [
            "where is",
            "where are",
            "locate",
            "location of",
            "find the",
            "find where",
            "point to",
            "show me where",
            "identify the location",
            "mark the",
            "bounding box",
            "bounding-box",
            "box around",
        ],
    }

    POINT_SIGNALS = [
        "point",
        "pixel",
        "coordinate",
        "coordinates",
        "marked point",
        "supplied point",
        "given point",
        "at x",
        "at (",
        "x=",
        "y=",
    ]

    REFERENCE_SIGNALS = [
        "largest",
        "smallest",
        "nearest",
        "closest",
        "urban area",
        "urban region",
        "building",
        "buildings",
        "road",
        "roads",
        "river",
        "water body",
        "waterbody",
        "forest",
        "agricultural",
        "agriculture",
        "pasture",
        "industrial",
        "commercial",
        "land use",
        "patch",
        "target",
        "object",
        "feature",
    ]

    PRIORITY = [
        "temporal_change",
        "cross_modal",
        "grounding",
        "single_image_vqa",
        "captioning",
    ]

    def _normalize(
        self,
        query: str,
    ) -> str:

        query = str(query).strip().lower()

        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        return query

    def _count_signals(
        self,
        query: str,
        signals,
    ) -> list:

        matches = []

        for signal in signals:
            if signal in query:
                matches.append(signal)

        return matches
    
    # -----------------------------------------------------------------
    # Single-image binary VQA eligibility
    # -----------------------------------------------------------------

    # These are semantic targets that occur in the remote-sensing
    # question space used by the frozen binary VQA specialist.
    #
    # This is intentionally a positive-support vocabulary rather than
    # a blacklist of individual unsupported objects.
    BINARY_TARGET_SIGNALS = [
        # Urban / built environment
        "urban",
        "urban fabric",
        "urban area",
        "urban areas",
        "urban region",
        "urban regions",
        "city",
        "cities",
        "town",
        "towns",
        "settlement",
        "settlements",
        "built-up",
        "built up",
        "built-up area",
        "built-up areas",
        "built up area",
        "built up areas",
        "residential",
        "residential area",
        "residential areas",
        "commercial",
        "commercial area",
        "commercial areas",
        "industrial",
        "industrial area",
        "industrial areas",
        "building",
        "buildings",
        "structure",
        "structures",
        "built structure",
        "built structures",
        "house",
        "houses",

        # Forest / trees
        "forest",
        "forests",
        "mixed forest",
        "mixed forests",
        "broad-leaved forest",
        "broad-leaved forests",
        "broad leaved forest",
        "broad leaved forests",
        "coniferous forest",
        "coniferous forests",
        "woodland",
        "woodlands",
        "wooded",
        "tree",
        "trees",
        "tree cover",
        "tree canopy",
        "canopy",

        # Agriculture / cultivation
        "agriculture",
        "agricultural",
        "agricultural area",
        "agricultural areas",
        "farmland",
        "farmlands",
        "cropland",
        "croplands",
        "crop",
        "crops",
        "cultivated land",
        "cultivated area",
        "cultivated areas",
        "arable land",
        "field",
        "fields",
        "permanent crop",
        "permanent crops",
        "pasture",
        "pastures",
        "grazing land",
        "grassland",
        "grasslands",
        "meadow",
        "meadows",
        "orchard",
        "orchards",
        "vineyard",
        "vineyards",

        # Vegetation
        "vegetation",
        "vegetative cover",
        "plant cover",
        "plantation",
        "plantations",
        "low vegetation",
        "shrub",
        "shrubs",
        "shrubland",
        "grass",
        "green area",
        "green areas",
        "greenery",

        # Water
        "water",
        "water body",
        "water bodies",
        "waterbody",
        "waterbodies",
        "river",
        "rivers",
        "lake",
        "lakes",
        "stream",
        "streams",
        "reservoir",
        "reservoirs",
        "pond",
        "ponds",
        "wetland",
        "wetlands",
        "marsh",
        "marshes",

        # Transport / linear infrastructure
        "road",
        "roads",
        "roadway",
        "roadways",
        "highway",
        "highways",
        "street",
        "streets",
        "railway",
        "railways",
        "railroad",
        "railroads",
        "track",
        "tracks",
        "parking area",
        "parking areas",

        # Land / surface
        "land cover",
        "land-cover",
        "land use",
        "land-use",
        "bare land",
        "bare soil",
        "exposed soil",
        "exposed ground",
        "soil",
        "sand",
        "rock",
        "rocky area",
        "surface type",
        "surface",

        # Common remote-sensing wording
        "area",
        "region",
        "patch",
        "patches",
    ]

    BINARY_INTENT_PATTERNS = {
        "presence": [
            "is there",
            "are there",
            "is any",
            "are any",
            "is there any",
            "are there any",
            "does the image contain",
            "does this image contain",
            "do the image contain",
            "does the image show",
            "does this image show",
            "can you detect",
            "can .* detect",
            "can .* be detected",
            "can you find",
            "can .* be found",
            "is there evidence of",
            "presence of",
            "is .* visible",
            "are .* visible",
            "is .* present",
            "are .* present",
            "do .* appear",
            "does .* appear",
            "can .* be identified",
        ],

        "count": [
            "how many",
            "number of",
            "count of",
            "what is the number of",
            "how many are there",
            "how many can be seen",
            "how many are visible",
            "how many do you see",
        ],

        "area": [
            "area",
            "areas covered",
            "cover",
            "covers",
            "covered",
            "coverage",
            "extent",
            "occupy",
            "occupies",
            "percentage",
            "percent",
            "fraction",
            "m2",
            "m²",
            "km2",
            "km²",
            "square meter",
            "square meters",
            "square metre",
            "square metres",
            "hectare",
            "hectares",
            "ha",
            "how much of the image",
            "how much of this image",
        ],

        "adjacency": [
            "touch",
            "touching",
            "adjacent",
            "adjacent to",
            "next to",
            "beside",
            "alongside",
            "border",
            "bordering",
            "abutting",
            "neighbor",
            "neighbors",
            "neighbour",
            "neighbours",
            "connected to",
            "in contact with",
            "meet",
            "meets",
            "intersect",
            "intersects",
            "contiguous with",
        ],
    }

    def _check_binary_question_eligibility(
        self,
        query: str,
    ) -> dict:
        """
        Validate whether a single-image binary question belongs to the
        supported remote-sensing question space.

        A question must contain:
          1. a recognizable binary intent
          2. a supported Earth-observation target/concept

        This intentionally avoids object-specific blacklists.
        """

        normalized = self._normalize(query)

        matched_intents = {}

        for intent, patterns in self.BINARY_INTENT_PATTERNS.items():

            matches = []

            for pattern in patterns:

                if ".*" in pattern:
                    if re.search(
                        pattern,
                        normalized,
                    ):
                        matches.append(pattern)

                elif pattern in normalized:
                    matches.append(pattern)

            if matches:
                matched_intents[intent] = matches

        matched_targets = self._count_signals(
            normalized,
            self.BINARY_TARGET_SIGNALS,
        )

        if not matched_intents:
            raise ValueError(
                "Unsupported single-image binary question. "
                "Supported question forms cover presence, count, "
                "area, and adjacency queries about recognized "
                "remote-sensing features."
            )

        if not matched_targets:
            raise ValueError(
                "Unsupported single-image binary question. "
                "Please ask about a recognized remote-sensing "
                "feature such as land cover, vegetation, forest, "
                "water, agriculture, buildings, roads, or related "
                "scene elements."
            )

        return {
            "supported": True,
            "intents": list(matched_intents.keys()),
            "matched_intent_patterns": matched_intents,
            "matched_targets": matched_targets,
        }

    def validate_question_eligibility(
        self,
        analysis: QueryAnalysis,
        query: str,
        options=None,
    ) -> dict:
        """
        Public eligibility contract used before specialist execution.

        MCQ requests are deliberately excluded because the presence of
        four options already identifies the dedicated MCQ pathway.
        """

        if analysis.task != "single_image_vqa":
            return {
                "supported": True,
                "reason": "non-single-image-vqa task",
            }

        if options is not None:
            return {
                "supported": True,
                "reason": "MCQ request",
            }

        return self._check_binary_question_eligibility(
            query
        )

    def analyze(
        self,
        query: str,
        point=None,
        input_modalities=None,
    ) -> QueryAnalysis:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        normalized = self._normalize(query)

        if not normalized:
            raise ValueError(
                "Query cannot be empty."
            )

        scores: Dict[str, int] = {}
        matches_by_task: Dict[str, list] = {}

        for task, signals in self.TASK_SIGNAL_GROUPS.items():

            matches = self._count_signals(
                normalized,
                signals,
            )

            matches_by_task[task] = matches
            scores[task] = len(matches)

        # Explicit point input strengthens grounding.
        if point is not None:

            scores["grounding"] += 2

            matches_by_task["grounding"].append(
                "explicit point input"
            )

        # Select using deterministic priority for ties.
        best_task = max(
            self.PRIORITY,
            key=lambda task: (
                scores[task],
                -self.PRIORITY.index(task),
            ),
        )

        best_score = scores[best_task]

        # -----------------------------------------------------------------
        # Input-aware cross-modal routing
        # -----------------------------------------------------------------
        #
        # If the caller explicitly states that both S1 and S2 are present,
        # a VQA request should use the dedicated cross-modal specialist.
        #
        # Grounding and temporal tasks retain their existing priority because
        # they represent different task types rather than ordinary VQA.
        #
        paired_s1_s2 = False

        if input_modalities is not None:

            normalized_modalities = {
                str(modality).strip().lower()
                for modality in input_modalities
            }

            paired_s1_s2 = (
                "s1" in normalized_modalities
                and "s2" in normalized_modalities
            )

        if (
            paired_s1_s2
            and (
                best_score == 0
                or best_task in {
                    "single_image_vqa",
                    "cross_modal",
                }
            )
        ):

            best_task = "cross_modal"

            confidence = 1.0

            matched_signals = (
                matches_by_task["cross_modal"].copy()
            )

            if "paired S1+S2 input" not in matched_signals:
                matched_signals.append(
                    "paired S1+S2 input"
                )

        # No observable signal.
        elif best_score == 0:

            best_task = "single_image_vqa"

            confidence = 0.20

            matched_signals = []

        else:

            ordered_scores = sorted(
                scores.values(),
                reverse=True,
            )

            second_score = (
                ordered_scores[1]
                if len(ordered_scores) > 1
                else 0
            )

            confidence = (
                best_score
                / max(
                    best_score + second_score,
                    1,
                )
            )

            matched_signals = matches_by_task[
                best_task
            ].copy()

        grounding_mode = None

        if best_task == "grounding":

            point_matches = self._count_signals(
                normalized,
                self.POINT_SIGNALS,
            )

            reference_matches = self._count_signals(
                normalized,
                self.REFERENCE_SIGNALS,
            )

            if point is not None:

                grounding_mode = "point"

                matched_signals.append(
                    "point-mode evidence"
                )

            elif len(point_matches) > len(reference_matches):

                grounding_mode = "point"

                matched_signals.extend(
                    point_matches
                )

            else:

                grounding_mode = "reference"

                matched_signals.extend(
                    reference_matches
                )

            mode_signal_count = max(
                len(point_matches),
                len(reference_matches),
            )

            if mode_signal_count == 0:
                confidence *= 0.75

        return QueryAnalysis(
            task=best_task,
            grounding_mode=grounding_mode,
            confidence=float(
                min(
                    max(confidence, 0.0),
                    1.0,
                )
            ),
            matched_signals=matched_signals,
            normalized_query=normalized,
        )


# =============================================================================
# 4. MAIN SATQUERYAI AGENT
# =============================================================================

class SatQueryAIAgent:
    """
    Main orchestration layer for SatQueryAI.

    The agent intentionally does not know the internal implementation details
    of any specialist. It only knows the capability contract.
    """

    def __init__(
        self,
        registry: SatQueryAICapabilityRegistry,
        analyzer: SatQueryAIQueryAnalyzer,
    ) -> None:

        if not isinstance(
            registry,
            SatQueryAICapabilityRegistry,
        ):
            raise TypeError(
                "registry must be a "
                "SatQueryAICapabilityRegistry."
            )

        if not isinstance(
            analyzer,
            SatQueryAIQueryAnalyzer,
        ):
            raise TypeError(
                "analyzer must be a "
                "SatQueryAIQueryAnalyzer."
            )

        self.registry = registry
        self.analyzer = analyzer

    def analyze_query(
        self,
        query: str,
        point=None,
        input_modalities=None,
    ) -> QueryAnalysis:

        return self.analyzer.analyze(
            query=query,
            point=point,
            input_modalities=input_modalities,
        )
        
    def validate_question_eligibility(
        self,
        analysis: QueryAnalysis,
        query: str,
        options=None,
    ) -> dict:

        return self.analyzer.validate_question_eligibility(
            analysis=analysis,
            query=query,
            options=options,
        )

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------

    def _require(
        self,
        inputs: Dict[str, Any],
        name: str,
    ) -> None:

        if inputs.get(name) is None:
            raise ValueError(
                f"Required input missing: {name}"
            )

    def _validate_inputs(
        self,
        analysis: QueryAnalysis,
        inputs: Dict[str, Any],
    ) -> None:
        """
        Validate only what the selected capability requires.

        Grounding accepts either one modality or a paired S1+S2 input.
        Single-image VQA requires exactly one visual modality.

        This prevents the application layer from imposing paired-input
        assumptions on true single-modality specialists.
        """

        task = analysis.task

        if task == "grounding":

            s1_present = inputs.get("s1") is not None
            s2_present = inputs.get("s2") is not None

            if not s1_present and not s2_present:
                raise ValueError(
                    "Grounding requires at least one visual modality: s1 or s2."
                )

            self._require(
                inputs,
                "text_embedding",
            )

            if analysis.grounding_mode == "point":

                self._require(
                    inputs,
                    "point",
                )

        elif task == "single_image_vqa":

            s1_present = inputs.get("s1") is not None
            s2_present = inputs.get("s2") is not None

            if s1_present == s2_present:
                if s1_present:
                    raise ValueError(
                        "single_image_vqa requires exactly one visual modality;"
                        " received both s1 and s2."
                    )

                raise ValueError(
                    "single_image_vqa requires exactly one visual modality:"
                    " s1 or s2."
                )

            self._require(
                inputs,
                "text_embedding",
            )

        elif task == "temporal_change":

            self._require(
                inputs,
                "t1",
            )

            self._require(
                inputs,
                "t2",
            )

            self._require(
                inputs,
                "diff",
            )

            self._require(
                inputs,
                "text_embedding",
            )

        elif task == "cross_modal":

            self._require(
                inputs,
                "s1",
            )

            self._require(
                inputs,
                "s2",
            )

            self._require(
                inputs,
                "text_embedding",
            )

        elif task == "captioning":

            # Captioning implementation will define its own required inputs.
            pass

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def execute(
        self,
        query: str,
        input_modalities=None,
        **inputs: Any,
    ) -> Dict[str, Any]:

        total_start = time.perf_counter()

        analysis = self.analyze_query(
            query=query,
            point=inputs.get("point"),
            input_modalities=input_modalities,
        )

        # Reject unsupported single-image binary questions before any
        # specialist is allowed to execute.
        self.validate_question_eligibility(
            analysis=analysis,
            query=query,
            options=inputs.get("options"),
        )

        validation_start = time.perf_counter()

        capability_entry = self.registry.get(
            analysis.task
        )

        capability_available = (
            capability_entry is not None
        )

        selected_tool = (
            capability_entry["tool_name"]
            if capability_entry is not None
            else None
        )

        # ---------------------------------------------------------------------
        # Capability unavailable
        # ---------------------------------------------------------------------

        if not capability_available:

            validation_end = time.perf_counter()
            total_end = time.perf_counter()

            trace = AgentExecutionTrace(
                query=query,
                normalized_query=analysis.normalized_query,
                detected_task=analysis.task,
                grounding_mode=analysis.grounding_mode,
                selected_tool=None,
                capability_available=False,
                status="capability_unavailable",
                routing_confidence=analysis.confidence,
                matched_signals=analysis.matched_signals,
                validation_ms=(
                    validation_end
                    - validation_start
                ) * 1000.0,
                execution_ms=0.0,
                total_ms=(
                    total_end
                    - total_start
                ) * 1000.0,
            )

            return {
                "answer_status": (
                    "capability_unavailable"
                ),
                "analysis": asdict(analysis),
                "evidence": None,
                "execution_trace": asdict(trace),
            }

        # ---------------------------------------------------------------------
        # Validate selected capability
        # ---------------------------------------------------------------------

        self._validate_inputs(
            analysis,
            inputs,
        )

        validation_end = time.perf_counter()

        # ---------------------------------------------------------------------
        # Execute specialist
        # ---------------------------------------------------------------------

        execution_start = time.perf_counter()

        handler = capability_entry["handler"]

        result = handler(
            query=query,
            analysis=analysis,
            **inputs,
        )

        execution_end = time.perf_counter()
        total_end = time.perf_counter()

        trace = AgentExecutionTrace(
            query=query,
            normalized_query=analysis.normalized_query,
            detected_task=analysis.task,
            grounding_mode=analysis.grounding_mode,
            selected_tool=selected_tool,
            capability_available=True,
            status="success",
            routing_confidence=analysis.confidence,
            matched_signals=analysis.matched_signals,
            validation_ms=(
                validation_end
                - validation_start
            ) * 1000.0,
            execution_ms=(
                execution_end
                - execution_start
            ) * 1000.0,
            total_ms=(
                total_end
                - total_start
            ) * 1000.0,
        )

        return {
            "answer_status": "success",
            "analysis": asdict(analysis),
            "evidence": result,
            "execution_trace": asdict(trace),
        }


# =============================================================================
# 5. FACTORY
# =============================================================================

def create_satqueryai_agent() -> SatQueryAIAgent:
    """
    Create a clean agent instance with no specialists registered.

    Specialists are deliberately registered by the application bootstrap
    rather than at module import time.
    """

    registry = SatQueryAICapabilityRegistry()
    analyzer = SatQueryAIQueryAnalyzer()

    return SatQueryAIAgent(
        registry=registry,
        analyzer=analyzer,
    )