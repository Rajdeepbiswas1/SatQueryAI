from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.analyze_endpoint import router as analyze_router
from backend.preview_endpoint import router as preview_router
from backend.auth_endpoint import router as auth_router
from backend.auth_service import (
    SESSION_COOKIE_NAME,
    verify_session_token,
)


app = FastAPI(
    title="SatQueryAI",
    version="1.0.0",
    description=(
        "Interactive Vision-Language Assistant "
        "for Multimodal Remote Sensing"
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PUBLIC_PATHS = {
    "/",
    "/health",
}


@app.middleware("http")
async def authentication_middleware(
    request: Request,
    call_next,
):
    """Require a valid session for non-public backend services."""

    path = request.url.path

    if (
        request.method == "OPTIONS"
        or path in PUBLIC_PATHS
        or path.startswith("/auth")
    ):
        return await call_next(request)

    token = request.cookies.get(
        SESSION_COOKIE_NAME
    )

    if verify_session_token(token) is None:
        return JSONResponse(
            status_code=401,
            content={
                "status": "unauthorized",
                "detail": "Authentication required.",
            },
        )

    return await call_next(request)


app.include_router(
    auth_router
)

app.include_router(
    analyze_router
)

app.include_router(
    preview_router
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "SatQueryAI",
        "message": "SatQueryAI backend is running.",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }
