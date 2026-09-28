from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from backend.auth_service import (
    SESSION_COOKIE_NAME,
    SESSION_DURATION_SECONDS,
    create_session_token,
    verify_credentials,
    verify_session_token,
)


router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(
    payload: LoginRequest,
    response: Response,
) -> dict[str, str]:

    if not verify_credentials(
        payload.username,
        payload.password,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    token = create_session_token(
        payload.username.strip()
    )

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_DURATION_SECONDS,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )

    return {
        "status": "success",
        "username": payload.username.strip(),
    }


@router.get("/me")
async def me(
    request: Request,
) -> dict[str, str]:

    payload = verify_session_token(
        request.cookies.get(SESSION_COOKIE_NAME)
    )

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated.",
        )

    return {
        "status": "authenticated",
        "username": str(payload["username"]),
    }


@router.post("/logout")
async def logout(
    response: Response,
) -> dict[str, str]:

    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
    )

    return {
        "status": "success",
    }
