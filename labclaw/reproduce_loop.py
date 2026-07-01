"""Karpathy-style autoresearch loop for reproducing paper claims."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from labclaw.eval_harness import ExperimentSpec, MetricResult, default_registry, metric_value

AttemptBuilder = Callable[[int, list[dict[str, Any]]], tuple[str, str]]


@dataclass
class ReproduceAttempt:
    attempt: int
    baseline_command: str
    candidate_command: str
    metric_result: dict[str, Any]
    kept: bool
    rationale: str


@dataclass
class ReproduceJournal:
    claim_id: str
    metric: str
    threshold: float
    attempts: list[ReproduceAttempt] = field(default_factory=list)

    def append(self, attempt: ReproduceAttempt) -> None:
        self.attempts.append(attempt)

    def best(self) -> ReproduceAttempt | None:
        improved = [item for item in self.attempts if item.kept]
        if not improved:
            return None
        return max(improved, key=lambda item: item.metric_result.get("delta") or 0)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["best_attempt"] = asdict(self.best()) if self.best() else None
        return payload


def default_attempt_builder(
    attempt: int,
    history: list[dict[str, Any]],
    *,
    baseline: float,
    candidate: float,
    step: float = 0.5,
) -> tuple[str, str]:
    """Deterministic search: nudge candidate metric down each failed attempt."""
    adjusted = candidate if attempt <= 1 else max(baseline + 0.1, candidate - step * (attempt - 1))
    return (
        f"metric:tokens_per_second={baseline}",
        f"metric:tokens_per_second={adjusted}",
    )


def run_reproduce_loop(
    *,
    claim_id: str,
    baseline: float,
    candidate: float,
    metric: str = "tokens_per_second",
    threshold: float | None = None,
    max_attempts: int = 5,
    builder: AttemptBuilder | None = None,
) -> ReproduceJournal:
    """Try bounded metric commands until the claim reproduces or attempts exhaust."""
    threshold = threshold if threshold is not None else max(1.0, candidate - baseline)
    journal = ReproduceJournal(claim_id=claim_id, metric=metric, threshold=threshold)
    registry = default_registry()
    history: list[dict[str, Any]] = []
    builder = builder or (
        lambda attempt, hist: default_attempt_builder(
            attempt, hist, baseline=baseline, candidate=candidate
        )
    )

    for attempt in range(1, max_attempts + 1):
        baseline_command, candidate_command = builder(attempt, history)
        spec = ExperimentSpec(
            claim_id=claim_id,
            cluster_id="reproduce-loop",
            harness="tiny_metric",
            baseline_command=baseline_command,
            candidate_command=candidate_command,
            metric=metric,
            direction="higher_is_better",
            threshold=threshold,
            metadata={"attempt": attempt},
        )
        result = registry.run(spec)
        kept = bool(result.improved)
        entry = ReproduceAttempt(
            attempt=attempt,
            baseline_command=baseline_command,
            candidate_command=candidate_command,
            metric_result=result.to_dict(),
            kept=kept,
            rationale="threshold met" if kept else "below threshold",
        )
        journal.append(entry)
        history.append(entry.metric_result)
        if kept:
            break
        time.sleep(0)
    return journal


def parse_program_md(path: Path) -> dict[str, Any]:
    """Minimal program.md parser for reproduce-loop configuration."""
    text = path.read_text(encoding="utf-8")
    config: dict[str, Any] = {"max_attempts": 5, "metric": "tokens_per_second"}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"^(\w+):\s*(.+)$", line.strip())
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if key in {"baseline", "candidate", "threshold", "max_attempts"}:
            config[key] = float(value) if key != "max_attempts" else int(float(value))
        else:
            config[key] = value
    return config


def persist_journal(journal: ReproduceJournal, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(journal.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
