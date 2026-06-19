import base64
import logging
import os
import re
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_SECONDS = 20.0
GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={api_key}"
)

_ALLOWED_LABELS = {"fake", "real"}
_MIME_BY_PIL_FORMAT = {
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


class GeminiServiceError(RuntimeError):
    """Raised when Gemini request/response processing fails."""


class GeminiConfigurationError(GeminiServiceError):
    """Raised when required Gemini configuration is missing."""


class GeminiUnexpectedResponseError(GeminiServiceError):
    """Raised when Gemini response cannot be mapped to fake/real safely."""


def _extract_text_from_response(payload: dict[str, Any]) -> str:
    """Extract text from top Gemini candidate with robust missing-parts handling."""
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        logger.error(
            "Gemini response missing candidates: raw_response_body=%s",
            payload,
        )
        raise GeminiUnexpectedResponseError(
            "Gemini response did not contain candidates output."
        )

    try:
        first_candidate = candidates[0]
        finish_reason = first_candidate.get("finishReason", "UNKNOWN")

        content = first_candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None

        text: str | None = None
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text = part["text"]
                    break

        if text:
            logger.info(
                "Gemini candidate parsed successfully: finishReason=%s",
                finish_reason,
            )
            return text

        if finish_reason == "MAX_TOKENS":
            logger.error(
                "Gemini returned no text before token stop: finishReason=%s candidate_structure=%s raw_response_body=%s",
                finish_reason,
                first_candidate,
                payload,
            )
            raise GeminiUnexpectedResponseError(
                "Gemini stopped at MAX_TOKENS without text output."
            )

        logger.error(
            "Gemini response missing text parts: finishReason=%s candidate_structure=%s raw_response_body=%s",
            finish_reason,
            first_candidate,
            payload,
        )
        raise GeminiUnexpectedResponseError(
            "Gemini response did not contain expected text output."
        )
    except (IndexError, KeyError, TypeError) as exc:
        logger.error(
            "Gemini response parse exception: raw_response_body=%s",
            payload,
        )
        raise GeminiUnexpectedResponseError(
            "Gemini response did not contain expected text output."
        ) from exc


def _sanitize_binary_label(raw_text: str) -> str:
    """Normalize Gemini text and only accept exact binary labels."""
    cleaned = raw_text.strip().lower()

    if cleaned in _ALLOWED_LABELS:
        return cleaned

    # Accept phrases only if they contain one unambiguous target token.
    tokens = set(re.findall(r"\b(fake|real)\b", cleaned))
    if len(tokens) == 1:
        return next(iter(tokens))

    raise GeminiUnexpectedResponseError(
        "Gemini output was not a safe single fake/real decision."
    )


def _detect_image_mime_type(image_bytes: bytes) -> str:
    """Infer MIME type from image bytes for Gemini inlineData."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = (image.format or "").lower()
    except (OSError, ValueError) as exc:
        raise GeminiServiceError("Unable to detect image format for Gemini request.") from exc

    return _MIME_BY_PIL_FORMAT.get(image_format, "image/jpeg")


async def classify_image_with_gemini(image_bytes: bytes) -> str:
    """Send image bytes to Gemini and return a strict fake/real label."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error(
            "Gemini configuration failure: model=%s request_url=%s missing_api_key=%s",
            GEMINI_MODEL,
            GEMINI_API_URL_TEMPLATE.format(model=GEMINI_MODEL, api_key="<REDACTED>"),
            True,
        )
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is not configured on the backend environment."
        )

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_mime_type = _detect_image_mime_type(image_bytes)

    prompt = (
        "Return exactly one lowercase word: fake or real."
    )

    request_body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": image_mime_type,
                            "data": image_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 32,
            "topP": 1,
            "topK": 1,
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
        },
    }

    url = GEMINI_API_URL_TEMPLATE.format(model=GEMINI_MODEL, api_key=api_key)
    log_url = GEMINI_API_URL_TEMPLATE.format(model=GEMINI_MODEL, api_key="<REDACTED>")

    logger.info(
        "Gemini request start: model=%s request_url=%s missing_api_key=%s image_mime_type=%s image_bytes=%s",
        GEMINI_MODEL,
        log_url,
        False,
        image_mime_type,
        len(image_bytes),
    )

    try:
        async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=request_body)
            response.raise_for_status()
            logger.info(
                "Gemini request success: model=%s request_url=%s status_code=%s",
                GEMINI_MODEL,
                log_url,
                response.status_code,
            )
            payload = response.json()
    except httpx.TimeoutException as exc:
        logger.exception(
            "Gemini timeout failure: model=%s request_url=%s missing_api_key=%s response_parsing_failed=%s timeout_seconds=%.1f",
            GEMINI_MODEL,
            log_url,
            False,
            False,
            GEMINI_TIMEOUT_SECONDS,
        )
        raise GeminiServiceError("Gemini request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raw_body = exc.response.text
        logger.error(
            "Gemini HTTP failure: model=%s request_url=%s status_code=%s missing_api_key=%s response_parsing_failed=%s raw_response_body=%s",
            GEMINI_MODEL,
            log_url,
            exc.response.status_code,
            False,
            False,
            raw_body,
        )
        raise GeminiServiceError(
            f"Gemini returned HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception(
            "Gemini transport failure: model=%s request_url=%s missing_api_key=%s response_parsing_failed=%s",
            GEMINI_MODEL,
            log_url,
            False,
            False,
        )
        raise GeminiServiceError("Gemini request failed.") from exc
    except ValueError as exc:
        logger.exception(
            "Gemini JSON parse failure: model=%s request_url=%s missing_api_key=%s response_parsing_failed=%s",
            GEMINI_MODEL,
            log_url,
            False,
            True,
        )
        raise GeminiServiceError("Gemini returned invalid JSON.") from exc

    try:
        raw_text = _extract_text_from_response(payload)
    except GeminiUnexpectedResponseError as exc:
        logger.error(
            "Gemini response schema parse failure: model=%s request_url=%s missing_api_key=%s response_parsing_failed=%s raw_response_body=%s",
            GEMINI_MODEL,
            log_url,
            False,
            True,
            payload,
        )
        raise GeminiServiceError("Gemini response format was invalid.") from exc

    try:
        return _sanitize_binary_label(raw_text)
    except GeminiUnexpectedResponseError as exc:
        logger.error(
            "Gemini label parse failure: model=%s request_url=%s missing_api_key=%s response_parsing_failed=%s raw_response_body=%s",
            GEMINI_MODEL,
            log_url,
            False,
            True,
            raw_text,
        )
        raise GeminiServiceError("Gemini returned an invalid binary label.") from exc
