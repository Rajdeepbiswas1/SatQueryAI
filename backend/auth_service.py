from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path


SESSION_COOKIE_NAME = "satquery_session"
SESSION_DURATION_SECONDS = 12 * 60 * 60

_USERNAME = os.getenv("SATQUERYAI_AUTH_USERNAME", "admin")
_PASSWORD = os.getenv("SATQUERYAI_AUTH_PASSWORD", "SatQueryAI@2026")

_SECRET_PATH = Path(__file__).with_name(".auth_secret")


def _load_or_create_secret() -> bytes:
    """Load a local signing secret, creating one on first run."""

    configured = os.getenv("SATQUERYAI_AUTH_SECRET")

    if configured:
        return configured.encode("utf-8")

    try:
        if _SECRET_PATH.is_file():
            value = _SECRET_PATH.read_text(encoding="utf-8").strip()
            if value:
                return value.encode("utf-8")

        value = secrets.token_urlsafe(48)
        _SECRET_PATH.write_text(value, encoding="utf-8")
        return value.encode("utf-8")

    except OSError as exc:
        raise RuntimeError(
            "Unable to initialize SatQueryAI authentication secret. "
            "Set SATQUERYAI_AUTH_SECRET explicitly."
        ) from exc


_SECRET = _load_or_create_secret()


def _password_digest(password: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        _SECRET,
        180_000,
    ).hex()


_CONFIGURED_PASSWORD_DIGEST = _password_digest(_PASSWORD)


def verify_credentials(username: str, password: str) -> bool:
    return hmac.compare_digest(
        username.strip(),
        _USERNAME,
    ) and hmac.compare_digest(
        _password_digest(password),
        _CONFIGURED_PASSWORD_DIGEST,
    )


def create_session_token(username: str) -> str:
    expires_at = int(time.time()) + SESSION_DURATION_SECONDS
    payload = {
        "username": username,
        "expires": expires_at,
        "nonce": secrets.token_hex(8),
    }

    payload_bytes = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    encoded_payload = base64.urlsafe_b64encode(
        payload_bytes
    ).decode("ascii").rstrip("=")

    signature = hmac.new(
        _SECRET,
        encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).digest()

    encoded_signature = base64.urlsafe_b64encode(
        signature
    ).decode("ascii").rstrip("=")

    return f"{encoded_payload}.{encoded_signature}"


def verify_session_token(token: str | None) -> dict[str, object] | None:
    if not token or "." not in token:
        return None

    encoded_payload, encoded_signature = token.split(".", 1)

    expected_signature = hmac.new(
        _SECRET,
        encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).digest()

    try:
        provided_signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        payload_bytes = base64.urlsafe_b64decode(
            encoded_payload + "=" * (-len(encoded_payload) % 4)
        )
    except (ValueError, UnicodeError):
        return None

    if not hmac.compare_digest(
        provided_signature,
        expected_signature,
    ):
        return None

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8")
        )
    except (ValueError, UnicodeError):
        return None

    username = payload.get("username")
    expires = payload.get("expires")

    if not isinstance(username, str):
        return None

    if username != _USERNAME:
        return None

    if not isinstance(expires, int):
        return None

    if expires <= int(time.time()):
        return None

    return payload
