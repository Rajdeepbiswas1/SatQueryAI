from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from agent.bootstrap import create_runtime_agent
from agent.response_formatter import format_agent_response
from agent.temporal_agent_service import TemporalAgentService
from backend.analyze_endpoint import router as analyze_router

# -------------------------------------------------------------------
# Application
# -------------------------------------------------------------------

app = FastAPI(
    title="SatQueryAI API",
    version="1.0.0",
    description="Backend API for the SatQueryAI remote-sensing assistant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)


# -------------------------------------------------------------------
# Lazy runtime agent
#
# The frozen agent/model stack is loaded only when first needed.
# -------------------------------------------------------------------

_runtime_agent = None
_temporal_service = None


def get_runtime_agent():
    global _runtime_agent

    if _runtime_agent is None:
        runtime = create_runtime_agent()
        _runtime_agent = runtime[0]

    return _runtime_agent


def get_temporal_service() -> TemporalAgentService:
    global _temporal_service

    if _temporal_service is None:
        _temporal_service = TemporalAgentService(
            get_runtime_agent()
        )

    return _temporal_service


# -------------------------------------------------------------------
# Basic endpoints
# -------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "SatQueryAI API",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "SatQueryAI API",
    }


@app.get("/agent/status")
def agent_status() -> dict[str, Any]:
    try:
        agent = get_runtime_agent()

        return {
            "status": "ok",
            "agent": "SatQueryAIAgent",
            "capabilities": agent.registry.capabilities(),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent initialization failed: {exc}",
        ) from exc


# -------------------------------------------------------------------
# Analysis endpoint
#
# Current general upload endpoint remains intentionally unchanged.
# Arbitrary uploaded images are not yet converted into the temporal
# specialist's required [196,768] feature representation here.
# -------------------------------------------------------------------

@app.post("/analyze-placeholder")
async def analyze(
    query: str = Form(...),
    image_1: UploadFile = File(...),
    image_2: UploadFile | None = File(None),
) -> dict[str, Any]:

    query = query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    image_1_bytes = await image_1.read()

    image_2_bytes = None

    if image_2 is not None:
        image_2_bytes = await image_2.read()

    if not image_1_bytes:
        raise HTTPException(
            status_code=400,
            detail="First image is empty.",
        )

    if image_2 is not None and not image_2_bytes:
        raise HTTPException(
            status_code=400,
            detail="Second image is empty.",
        )

    return {
        "status": "received",
        "query": query,
        "files": {
            "image_1": {
                "filename": image_1.filename,
                "content_type": image_1.content_type,
                "size_bytes": len(image_1_bytes),
            },
            "image_2": (
                {
                    "filename": image_2.filename,
                    "content_type": image_2.content_type,
                    "size_bytes": len(image_2_bytes),
                }
                if image_2 is not None
                else None
            ),
        },
        "message": (
            "Input received successfully. "
            "Image-to-feature processing will be connected next."
        ),
    }


# -------------------------------------------------------------------
# Verified Temporal Test endpoint
#
# This endpoint intentionally uses the verified held-out Test cache.
# It is an application integration endpoint, not an arbitrary-image
# inference endpoint.
# -------------------------------------------------------------------

@app.post("/temporal/test-question")
def temporal_test_question(
    question_id: int = Form(...),
) -> dict[str, Any]:

    if question_id < 0 or question_id >= 39_686:
        raise HTTPException(
            status_code=400,
            detail=(
                "question_id must be in the verified Test range "
                "0..39685."
            ),
        )

    try:
        service = get_temporal_service()

        response = service.execute_test_question(
            question_id=question_id
        )

        return response

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Temporal execution failed: {exc}",
        ) from exc