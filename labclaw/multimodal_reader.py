"""Extract testable ML/code claim cards with Gemma on Cerebras."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import mimetypes
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class SourceRecord:
    """Source scout contract consumed by the reader swarm."""

    source_id: str
    kind: str
    title: str
    raw_text: str
    url: str | None = None
    published_at: str | None = None
    figures: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReaderSource:
    """Normalized in-memory source accepted by the batch reader."""

    source_id: str
    title: str
    content: str
    path: Path | None = None
    figures: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReaderSourceResult:
    source_id: str
    title: str
    ok: bool
    elapsed_ms: float
    card_count: int
    result: ReaderResult | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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


def read_source_text(
    content: str,
    *,
    title: str = "untitled-source",
    source_path: Path | None = None,
    figures: list[dict[str, Any]] | None = None,
    use_gemma: bool = True,
    client: Any | None = None,
) -> ReaderResult:
    """Read source text supplied by scouts without requiring a file on disk."""
    if use_gemma:
        return extract_with_gemma(
            content,
            source_path=source_path,
            source_title=title,
            figure_refs=normalize_figure_refs(figures or markdown_figures(content)),
            client=client,
        )
    result = parse_local_fixture(content_with_figures(content, figures or []), source_path=source_path)
    return with_source_metadata(result, title=title, source_path=source_path)


def read_source_record(
    record: SourceRecord,
    *,
    use_gemma: bool = True,
    client: Any | None = None,
) -> ReaderResult:
    return read_source_text(
        record.raw_text,
        title=record.title,
        figures=record.figures,
        use_gemma=use_gemma,
        client=client,
    )


def read_sources(
    sources: Sequence[SourceRecord | ReaderSource | Path],
    *,
    max_workers: int = 4,
    use_gemma: bool = True,
    client: Any | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> list[ReaderSourceResult]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    indexed_sources = list(enumerate(sources))
    results: list[ReaderSourceResult | None] = [None] * len(indexed_sources)
    worker_count = min(max_workers, len(indexed_sources)) or 1

    def run_one(index: int, source: SourceRecord | ReaderSource | Path) -> tuple[int, ReaderSourceResult]:
        source_id, title, metadata = source_identity(source, index)
        started = time.perf_counter()
        try:
            source_client = client_factory() if client_factory else client
            result = read_source_input(source, use_gemma=use_gemma, client=source_client)
        except Exception as exc:  # noqa: BLE001 - failure isolation is the point of the batch API.
            elapsed_ms = (time.perf_counter() - started) * 1000
            return index, ReaderSourceResult(
                source_id=source_id,
                title=title,
                ok=False,
                elapsed_ms=elapsed_ms,
                card_count=0,
                error=f"{type(exc).__name__}: {exc}",
                metadata=metadata,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return index, ReaderSourceResult(
            source_id=source_id,
            title=title,
            ok=True,
            elapsed_ms=elapsed_ms,
            card_count=len(result.cards),
            result=result,
            metadata=metadata,
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_one, index, source) for index, source in indexed_sources]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result

    return [result for result in results if result is not None]


def read_source_input(
    source: SourceRecord | ReaderSource | Path,
    *,
    use_gemma: bool,
    client: Any | None,
) -> ReaderResult:
    if isinstance(source, SourceRecord):
        return read_source_record(source, use_gemma=use_gemma, client=client)
    if isinstance(source, ReaderSource):
        return read_source_text(
            source.content,
            title=source.title,
            source_path=source.path,
            figures=source.figures,
            use_gemma=use_gemma,
            client=client,
        )
    return read_source(Path(source), use_gemma=use_gemma, client=client)


def extract_with_gemma(
    content: str,
    *,
    source_path: Path | None = None,
    source_title: str | None = None,
    figure_refs: list[dict[str, Any]] | None = None,
    client: Any | None = None,
) -> ReaderResult:
    client = client or cerebras_client()
    figure_refs = normalize_figure_refs(figure_refs or markdown_figures(content))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_content(
                content,
                source_path=source_path,
                source_title=source_title,
                figure_refs=figure_refs,
            ),
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
    result = result_from_dict(load_json_object(raw))
    if source_title or source_path:
        return with_source_metadata(result, title=source_title, source_path=source_path)
    return result


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
    source_path: Path | None = None,
    source_title: str | None = None,
    figure_refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_label = str(source_path) if source_path else source_title or "in-memory source"
    payload: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Source: {source_label}\n\n"
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


def normalize_figure_refs(figures: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for figure in figures:
        path = figure.get("path") or figure.get("local_path") or figure.get("url")
        if not path:
            continue
        normalized.append(
            {
                "alt_text": str(figure.get("alt_text") or figure.get("caption") or ""),
                "path": str(path),
            }
        )
    return normalized


def content_with_figures(content: str, figures: list[dict[str, Any]]) -> str:
    refs = normalize_figure_refs(figures)
    if not refs:
        return content
    existing_paths = {figure["path"] for figure in markdown_figures(content)}
    extra_refs = [
        f"![{figure['alt_text']}]({figure['path']})"
        for figure in refs
        if figure["path"] not in existing_paths
    ]
    if not extra_refs:
        return content
    return f"{content.rstrip()}\n\n" + "\n".join(extra_refs)


def source_identity(
    source: SourceRecord | ReaderSource | Path,
    index: int,
) -> tuple[str, str, dict[str, Any]]:
    if isinstance(source, SourceRecord):
        return source.source_id, source.title, {"kind": source.kind, **source.metadata}
    if isinstance(source, ReaderSource):
        return source.source_id, source.title, source.metadata
    path = Path(source)
    return path.stem or f"source-{index + 1}", path.name, {"path": str(path)}


def with_source_metadata(
    result: ReaderResult,
    *,
    title: str | None,
    source_path: Path | None,
) -> ReaderResult:
    source = dict(result.source)
    if title:
        source["title"] = title
    if source_path:
        source["path"] = str(source_path)
    return ReaderResult(source=source, cards=result.cards)


def resolve_figure_path(source_path: Path | None, figure_path: str) -> Path | None:
    if figure_path.startswith(("http://", "https://", "data:")):
        return None
    path = Path(figure_path)
    if path.is_absolute():
        return path
    return source_path.parent / path if source_path else path


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


def batch_to_dict(results: Sequence[ReaderSourceResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


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
