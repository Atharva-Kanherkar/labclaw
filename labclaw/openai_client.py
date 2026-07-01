"""Shared OpenAI client helpers for structured JSON completions."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Protocol


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class StructuredJSONClient(Protocol):
    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        model: str,
    ) -> Mapping[str, Any] | str:
        ...


def openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def live_openai_enabled() -> bool:
    if not openai_api_key():
        return False
    return os.environ.get("LABCLAW_LIVE_OPENAI", "1") not in {"0", "false", "False"}


def load_json_object(raw: Mapping[str, Any] | str | None, *, provider: str = "OpenAI") -> dict[str, Any]:
    if raw is None:
        raise ValueError(f"{provider} returned no message content.")
    if isinstance(raw, Mapping):
        return dict(raw)
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{provider} returned malformed JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{provider} JSON response must be an object.")
    return parsed


class OpenAIClient:
    """Live OpenAI adapter with JSON-schema structured outputs."""

    def __init__(self, *, api_key: str | None = None, openai_client: Any | None = None) -> None:
        if openai_client is not None:
            self._client = openai_client
            return
        key = api_key or openai_api_key()
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY to use the live OpenAI client.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install `openai` to use the live OpenAI client.") from exc
        self._client = OpenAI(api_key=key)

    def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        model: str = DEFAULT_OPENAI_MODEL,
    ) -> Mapping[str, Any] | str:
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title", "structured_output"),
                    "strict": True,
                    "schema": schema,
                },
            },
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
