"""Extract testable ML/code claim cards with Gemma on Cerebras."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL = "gemma-4-31b"
MAX_IMAGES = 5
MAX_IMAGE_PAYLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg"}

IMAGE_PATTERN = re.compile(r"!\[([^\]]*)]\(([^)]+)\)")

SYSTEM_PROMPT = """You are LabClaw's multimodal reader.
Extract testable ML/code research claims from text and figures.
Prefer claims with runnable code, benchmark numbers, or figure evidence.
Do not invent numbers, commands, or figures."""

CLAIM_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "path": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["title", "path"],
            "additionalProperties": False,
        },
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "main_claim": {"type": "string"},
                    "figures": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "alt_text": {"type": "string"},
                                "path": {"type": "string"},
                                "caption": {"type": "string"},
                                "visual_observation": {"type": "string"},
                            },
                            "required": ["id", "alt_text", "path", "caption", "visual_observation"],
                            "additionalProperties": False,
                        },
                    },
                    "benchmark_numbers": {"type": "array", "items": {"type": "string"}},
                    "code_hooks": {"type": "array", "items": {"type": "string"}},
                    "is_testable": {"type": "boolean"},
                    "evidence_needed": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "id",
                    "main_claim",
                    "figures",
                    "benchmark_numbers",
                    "code_hooks",
                    "is_testable",
                    "evidence_needed",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["source", "cards"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class FigureReference:
    id: str
    alt_text: str
    path: str
    caption: str
    visual_observation: str = ""


@dataclass(frozen=True)
class ClaimCard:
    id: str
    main_claim: str
    figures: list[FigureReference]
    benchmark_numbers: list[str]
    code_hooks: list[str]
    is_testable: bool
    evidence_needed: list[str]


@dataclass(frozen=True)
class ReaderResult:
    source: dict[str, str | None]
    cards: list[ClaimCard]


def read_source(
    source_path: Path,
    *,
    use_gemma: bool = True,
    client: Any | None = None,
) -> ReaderResult:
    content = source_path.read_text(encoding="utf-8")
    if use_gemma:
        return extract_with_gemma(content, source_path=source_path, client=client)
    return parse_local_fixture(content, source_path=source_path)


def extract_with_gemma(
    content: str,
    *,
    source_path: Path,
    client: Any | None = None,
) -> ReaderResult:
    client = client or cerebras_client()
    figure_refs = markdown_figures(content)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_content(content, source_path=source_path, figure_refs=figure_refs),
        },
    ]
    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "claim_cards",
                "strict": True,
                "schema": CLAIM_CARD_SCHEMA,
            },
        },
        stream=False,
        max_completion_tokens=4096,
        temperature=0.2,
        top_p=1,
    )
    raw = response.choices[0].message.content
    return result_from_dict(load_json_object(raw))


def cerebras_client() -> Any:
    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError as exc:
        raise RuntimeError(
            "Install the Cerebras SDK with `pip install cerebras-cloud-sdk` "
            "or run with `--local-fixture` for offline tests."
        ) from exc

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("Set CEREBRAS_API_KEY or run with `--local-fixture`.")
    return Cerebras(api_key=api_key)


def build_user_content(
    content: str,
    *,
    source_path: Path,
    figure_refs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Source path: {source_path}\n\n"
                "Extract claim cards from this source. Inspect attached figures when present.\n\n"
                f"{content}"
            ),
        }
    ]

    image_count = 0
    image_payload_bytes = 0
    for figure in figure_refs:
        if image_count >= MAX_IMAGES:
            break
        image_path = resolve_figure_path(source_path, figure["path"])
        if not image_path or not image_path.exists():
            continue
        mime_type = mimetypes.guess_type(image_path)[0] or ""
        if mime_type not in SUPPORTED_IMAGE_TYPES:
            continue
        image_size = image_path.stat().st_size
        if image_payload_bytes + image_size > MAX_IMAGE_PAYLOAD_BYTES:
            continue
        image_payload_bytes += image_size
        image_count += 1
        payload.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_data_uri(image_path)},
            }
        )

    return payload


def markdown_figures(content: str) -> list[dict[str, str]]:
    return [
        {"alt_text": match.group(1).strip(), "path": match.group(2).strip()}
        for match in IMAGE_PATTERN.finditer(content)
    ]


def resolve_figure_path(source_path: Path, figure_path: str) -> Path | None:
    if figure_path.startswith(("http://", "https://", "data:")):
        return None
    path = Path(figure_path)
    return path if path.is_absolute() else source_path.parent / path


def encode_image_data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def load_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        raise ValueError("Gemma returned no message content.")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemma returned malformed JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Gemma JSON response must be an object.")
    return parsed


def result_from_dict(payload: dict[str, Any]) -> ReaderResult:
    cards = []
    for card in payload.get("cards", []):
        figures = [
            FigureReference(
                id=str(figure.get("id", f"figure-{index + 1}")),
                alt_text=str(figure.get("alt_text", "")),
                path=str(figure.get("path", "")),
                caption=str(figure.get("caption", "")),
                visual_observation=str(figure.get("visual_observation", "")),
            )
            for index, figure in enumerate(card.get("figures", []))
        ]
        cards.append(
            ClaimCard(
                id=str(card.get("id", f"claim-{len(cards) + 1}")),
                main_claim=str(card.get("main_claim", "")),
                figures=figures,
                benchmark_numbers=[str(value) for value in card.get("benchmark_numbers", [])],
                code_hooks=[str(value) for value in card.get("code_hooks", [])],
                is_testable=bool(card.get("is_testable", False)),
                evidence_needed=[str(value) for value in card.get("evidence_needed", [])],
            )
        )
    return ReaderResult(
        source={
            "title": str(payload.get("source", {}).get("title", "untitled-source")),
            "path": payload.get("source", {}).get("path"),
        },
        cards=cards,
    )


def parse_local_fixture(content: str, *, source_path: Path | None = None) -> ReaderResult:
    """Offline fallback for deterministic tests; production extraction uses Gemma."""
    title = next((line.removeprefix("# ").strip() for line in content.splitlines() if line.startswith("# ")), "untitled-source")
    figures = [
        FigureReference(
            id=f"figure-{index + 1}",
            alt_text=figure["alt_text"],
            path=figure["path"],
            caption="",
            visual_observation="",
        )
        for index, figure in enumerate(markdown_figures(content))
    ]
    numbers = re.findall(r"\b\d+(?:\.\d+)?\s?(?:x|%|seconds|tok/s|pass@1)\b", content, re.IGNORECASE)
    commands = [
        line.strip()
        for line in content.splitlines()
        if re.match(r"^\s*(?:git clone|python(?:3)?\s|pytest\b|pip install)", line, re.IGNORECASE)
    ]
    prose = " ".join(
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("![") and not line.startswith("```")
    )
    claim = next((sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", prose) if "claim" in sentence.lower()), "")
    return ReaderResult(
        source={"title": title, "path": str(source_path) if source_path else None},
        cards=[
            ClaimCard(
                id="claim-1",
                main_claim=claim,
                figures=figures,
                benchmark_numbers=list(dict.fromkeys(numbers)),
                code_hooks=list(dict.fromkeys(commands)),
                is_testable=bool(numbers and commands),
                evidence_needed=[
                    "run the listed code hook in the VM",
                    "compare produced metric/plot against referenced figure",
                ],
            )
        ],
    )


def to_dict(result: ReaderResult) -> dict[str, Any]:
    return asdict(result)


def format_human_result(result: ReaderResult) -> str:
    lines = [f"# {result.source['title']}"]
    if not result.cards:
        lines.append("No claims extracted.")
        return "\n".join(lines)

    for index, card in enumerate(result.cards, start=1):
        if len(result.cards) > 1:
            lines.append("")
            lines.append(f"## Claim {index}")
        lines.append(f"Claim: {card.main_claim or 'none'}")
        lines.append(f"Figures: {len(card.figures)}")
        lines.append(f"Benchmarks: {', '.join(card.benchmark_numbers) or 'none'}")
        lines.append(f"Code hooks: {len(card.code_hooks)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LabClaw multimodal claim cards with Gemma.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    parser.add_argument("--local-fixture", action="store_true", help="Skip Gemma and use deterministic local parsing.")
    args = parser.parse_args()

    try:
        result = read_source(args.source, use_gemma=not args.local_fixture)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.json:
        print(json.dumps(to_dict(result), indent=2))
        return

    print(format_human_result(result))


if __name__ == "__main__":
    main()
