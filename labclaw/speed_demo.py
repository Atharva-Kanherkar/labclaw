"""Cerebras reader swarm speed demo."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from labclaw.multimodal_reader import (
    ReaderSource,
    ReaderSourceResult,
    batch_to_dict,
    read_source_input,
    read_sources,
)


@dataclass(frozen=True)
class ProgressEvent:
    event: str
    lane: str
    source_id: str | None
    message: str
    elapsed_ms: float
    ok: bool | None = None


@dataclass(frozen=True)
class LaneSummary:
    name: str
    provider: str
    sources_completed: int
    elapsed_ms: float
    latency_ms_per_source: float
    estimated_tokens: int
    estimated_tokens_per_second: float
    claim_cards_produced: int
    errors: int
    simulated: bool


@dataclass(frozen=True)
class SpeedDemoReport:
    selected_cluster: str
    lanes: list[LaneSummary]
    progress_events: list[ProgressEvent]
    fast_results: list[ReaderSourceResult]
    baseline_results: list[ReaderSourceResult]


def load_fixture_batch(paths: Sequence[Path], *, repeat: int = 1) -> list[ReaderSource]:
    if repeat < 1:
        raise ValueError("repeat must be at least 1.")

    sources: list[ReaderSource] = []
    for repeat_index in range(repeat):
        for path in paths:
            content = path.read_text(encoding="utf-8")
            title = title_from_content(content) or path.stem
            sources.append(
                ReaderSource(
                    source_id=f"{path.stem}-{repeat_index + 1}",
                    title=title,
                    content=content,
                    path=path,
                    metadata={"cluster_id": "demo-reader-swarm", "fixture_path": str(path)},
                )
            )
    return sources


def run_speed_demo(
    sources: Sequence[ReaderSource],
    *,
    fast_workers: int = 4,
    baseline_delay_ms: float = 150,
    live_cerebras: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> SpeedDemoReport:
    progress_events: list[ProgressEvent] = []

    def emit(event: ProgressEvent) -> None:
        progress_events.append(event)
        if progress:
            progress(event)

    selected_cluster = select_demo_cluster(sources)
    fast_results, fast_summary = run_fast_lane(
        sources,
        max_workers=fast_workers,
        live_cerebras=live_cerebras,
        emit=emit,
    )
    baseline_results, baseline_summary = run_baseline_lane(
        sources,
        delay_ms=baseline_delay_ms,
        emit=emit,
    )
    return SpeedDemoReport(
        selected_cluster=selected_cluster,
        lanes=[fast_summary, baseline_summary],
        progress_events=progress_events,
        fast_results=fast_results,
        baseline_results=baseline_results,
    )


def run_fast_lane(
    sources: Sequence[ReaderSource],
    *,
    max_workers: int,
    live_cerebras: bool,
    emit: Callable[[ProgressEvent], None],
) -> tuple[list[ReaderSourceResult], LaneSummary]:
    lane = "cerebras-gemma-swarm"
    started = time.perf_counter()
    emit(ProgressEvent("lane_started", lane, None, "started Cerebras/Gemma reader swarm", 0.0))
    results = read_sources(sources, max_workers=max_workers, use_gemma=live_cerebras)
    elapsed_ms = (time.perf_counter() - started) * 1000
    for result in results:
        emit(
            ProgressEvent(
                "source_completed",
                lane,
                result.source_id,
                "source completed" if result.ok else result.error or "source failed",
                result.elapsed_ms,
                ok=result.ok,
            )
        )
    summary = summarize_lane(
        name=lane,
        provider="cerebras/gemma-4-31b",
        sources=sources,
        results=results,
        elapsed_ms=elapsed_ms,
        simulated=not live_cerebras,
    )
    emit(ProgressEvent("lane_finished", lane, None, "finished Cerebras/Gemma reader swarm", elapsed_ms))
    return results, summary


def run_baseline_lane(
    sources: Sequence[ReaderSource],
    *,
    delay_ms: float,
    emit: Callable[[ProgressEvent], None],
) -> tuple[list[ReaderSourceResult], LaneSummary]:
    lane = "simulated-slower-baseline"
    started = time.perf_counter()
    emit(ProgressEvent("lane_started", lane, None, "started simulated baseline", 0.0))
    results = []
    for source in sources:
        source_started = time.perf_counter()
        try:
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
            result = read_source_input(source, use_gemma=False, client=None)
        except Exception as exc:  # noqa: BLE001 - demo baseline should keep reporting.
            elapsed = (time.perf_counter() - source_started) * 1000
            source_result = ReaderSourceResult(
                source_id=source.source_id,
                title=source.title,
                ok=False,
                elapsed_ms=elapsed,
                card_count=0,
                error=f"{type(exc).__name__}: {exc}",
                metadata=source.metadata,
            )
        else:
            elapsed = (time.perf_counter() - source_started) * 1000
            source_result = ReaderSourceResult(
                source_id=source.source_id,
                title=source.title,
                ok=True,
                elapsed_ms=elapsed,
                card_count=len(result.cards),
                result=result,
                metadata=source.metadata,
            )
        results.append(source_result)
        emit(
            ProgressEvent(
                "source_completed",
                lane,
                source.source_id,
                "source completed" if source_result.ok else source_result.error or "source failed",
                source_result.elapsed_ms,
                ok=source_result.ok,
            )
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    summary = summarize_lane(
        name=lane,
        provider="simulated-baseline",
        sources=sources,
        results=results,
        elapsed_ms=elapsed_ms,
        simulated=True,
    )
    emit(ProgressEvent("lane_finished", lane, None, "finished simulated baseline", elapsed_ms))
    return results, summary


def summarize_lane(
    *,
    name: str,
    provider: str,
    sources: Sequence[ReaderSource],
    results: Sequence[ReaderSourceResult],
    elapsed_ms: float,
    simulated: bool,
) -> LaneSummary:
    completed = sum(1 for result in results if result.ok)
    cards = sum(result.card_count for result in results)
    errors = sum(1 for result in results if not result.ok)
    estimated_tokens = sum(estimate_tokens(source.content) for source in sources)
    elapsed_seconds = max(elapsed_ms / 1000, 0.001)
    return LaneSummary(
        name=name,
        provider=provider,
        sources_completed=completed,
        elapsed_ms=elapsed_ms,
        latency_ms_per_source=elapsed_ms / max(len(sources), 1),
        estimated_tokens=estimated_tokens,
        estimated_tokens_per_second=estimated_tokens / elapsed_seconds,
        claim_cards_produced=cards,
        errors=errors,
        simulated=simulated,
    )


def select_demo_cluster(sources: Sequence[ReaderSource]) -> str:
    for source in sources:
        cluster_id = source.metadata.get("cluster_id")
        if cluster_id:
            return str(cluster_id)
    return "demo-reader-swarm"


def estimate_tokens(content: str) -> int:
    return max(1, len(content) // 4)


def title_from_content(content: str) -> str | None:
    return next(
        (line.removeprefix("# ").strip() for line in content.splitlines() if line.startswith("# ")),
        None,
    )


def format_timing_table(report: SpeedDemoReport) -> str:
    lines = [
        f"Selected cluster: {report.selected_cluster}",
        "",
        "| Lane | Provider | Sources | Elapsed ms | Latency/source ms | Est tok/s | Claim cards | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in report.lanes:
        lines.append(
            "| "
            f"{lane.name} | "
            f"{lane.provider}{' (simulated)' if lane.simulated else ''} | "
            f"{lane.sources_completed} | "
            f"{lane.elapsed_ms:.1f} | "
            f"{lane.latency_ms_per_source:.1f} | "
            f"{lane.estimated_tokens_per_second:.1f} | "
            f"{lane.claim_cards_produced} | "
            f"{lane.errors} |"
        )
    return "\n".join(lines)


def progress_json_lines(events: Sequence[ProgressEvent]) -> str:
    return "\n".join(json.dumps(asdict(event), sort_keys=True) for event in events)


def report_to_dict(report: SpeedDemoReport) -> dict[str, Any]:
    return {
        "selected_cluster": report.selected_cluster,
        "lanes": [asdict(lane) for lane in report.lanes],
        "progress_events": [asdict(event) for event in report.progress_events],
        "fast_results": batch_to_dict(report.fast_results),
        "baseline_results": batch_to_dict(report.baseline_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LabClaw Cerebras reader swarm speed demo.")
    parser.add_argument("sources", type=Path, nargs="+", help="Markdown fixture sources to read.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat fixture sources to form a batch.")
    parser.add_argument("--fast-workers", type=int, default=4, help="Reader swarm concurrency.")
    parser.add_argument("--baseline-delay-ms", type=float, default=150, help="Delay per baseline source.")
    parser.add_argument("--live-cerebras", action="store_true", help="Use live Cerebras/Gemma instead of fixture parsing.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable report JSON.")
    parser.add_argument("--progress-jsonl", action="store_true", help="Print progress events as JSON lines before the report.")
    args = parser.parse_args()

    sources = load_fixture_batch(args.sources, repeat=args.repeat)
    report = run_speed_demo(
        sources,
        fast_workers=args.fast_workers,
        baseline_delay_ms=args.baseline_delay_ms,
        live_cerebras=args.live_cerebras,
    )
    if args.progress_jsonl:
        print(progress_json_lines(report.progress_events))
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))
        return
    print(format_timing_table(report))


if __name__ == "__main__":
    main()
