"""Eval harness registry and normalized metric outputs for LabClaw."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

HarnessRunner = Callable[["ExperimentSpec"], "MetricResult"]


@dataclass(frozen=True)
class ExperimentSpec:
    claim_id: str
    cluster_id: str
    harness: str
    baseline_command: str
    candidate_command: str
    metric: str
    direction: str
    threshold: float
    threshold_mode: str = "absolute_delta"
    goal: str = ""
    source_id: str | None = None
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricResult:
    claim_id: str
    cluster_id: str
    harness: str
    metric: str
    direction: str
    threshold: float
    threshold_mode: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    improved: bool
    status: str
    artifacts: list[str] = field(default_factory=list)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HarnessRegistry:
    def __init__(self) -> None:
        self._harnesses: dict[str, HarnessRunner] = {}

    def register(self, name: str, runner: HarnessRunner) -> None:
        if not name:
            raise ValueError("Harness name cannot be empty.")
        self._harnesses[name] = runner

    def run(self, spec: ExperimentSpec) -> MetricResult:
        try:
            runner = self._harnesses[spec.harness]
        except KeyError as exc:
            raise KeyError(f"Unknown eval harness: {spec.harness}") from exc
        return runner(spec)


def spec_from_pi_proposal(
    proposal: Mapping[str, Any],
    *,
    harness: str = "tiny_metric",
    direction: str = "higher_is_better",
    threshold: float | None = None,
    source_id: str | None = None,
) -> ExperimentSpec:
    """Build an ExperimentSpec from #13's Gemini PI experiment_proposal."""
    if not bool(proposal.get("should_run", False)):
        raise ValueError("PI proposal declined the experiment; no ExperimentSpec created.")
    required = ["claim_id", "cluster_id", "baseline_command", "candidate_command", "metric", "threshold"]
    missing = [key for key in required if not proposal.get(key)]
    if missing:
        raise ValueError(f"PI proposal missing required experiment field(s): {', '.join(missing)}")
    threshold_mode, threshold_value = parse_threshold(proposal["threshold"])
    if threshold is not None:
        threshold_mode = "absolute_delta"
        threshold_value = threshold
    return ExperimentSpec(
        claim_id=str(proposal["claim_id"]),
        cluster_id=str(proposal["cluster_id"]),
        harness=harness,
        baseline_command=str(proposal["baseline_command"]),
        candidate_command=str(proposal["candidate_command"]),
        metric=str(proposal["metric"]),
        direction=direction,
        threshold=threshold_value,
        threshold_mode=threshold_mode,
        goal=str(proposal.get("goal", "")),
        source_id=source_id,
        metadata={"pi_threshold": str(proposal["threshold"]), "pi_rationale": str(proposal.get("rationale", ""))},
    )


def tiny_metric_harness(spec: ExperimentSpec) -> MetricResult:
    try:
        baseline = metric_value(spec.baseline_command)
        candidate = metric_value(spec.candidate_command)
    except ValueError as exc:
        return MetricResult(
            claim_id=spec.claim_id,
            cluster_id=spec.cluster_id,
            harness=spec.harness,
            metric=spec.metric,
            direction=spec.direction,
            threshold=spec.threshold,
            threshold_mode=spec.threshold_mode,
            baseline=None,
            candidate=None,
            delta=None,
            improved=False,
            status="failed",
            artifacts=list(spec.artifacts),
            failure_reason=str(exc),
        )

    delta = candidate - baseline
    if spec.direction == "lower_is_better":
        delta = baseline - candidate
    elif spec.direction != "higher_is_better":
        raise ValueError(f"Unknown metric direction: {spec.direction}")

    if spec.threshold_mode == "absolute_delta":
        improved = delta >= spec.threshold
    elif spec.threshold_mode == "relative_ratio":
        if baseline == 0:
            return MetricResult(
                claim_id=spec.claim_id,
                cluster_id=spec.cluster_id,
                harness=spec.harness,
                metric=spec.metric,
                direction=spec.direction,
                threshold=spec.threshold,
                threshold_mode=spec.threshold_mode,
                baseline=baseline,
                candidate=candidate,
                delta=delta,
                improved=False,
                status="failed",
                artifacts=list(spec.artifacts),
                failure_reason="Relative threshold cannot compare against zero baseline.",
            )
        ratio = candidate / baseline
        improved = ratio <= spec.threshold if spec.direction == "lower_is_better" else ratio >= spec.threshold
    else:
        raise ValueError(f"Unknown threshold mode: {spec.threshold_mode}")

    status = "improved" if improved else "no_change"
    if delta < 0:
        status = "worse"

    return MetricResult(
        claim_id=spec.claim_id,
        cluster_id=spec.cluster_id,
        harness=spec.harness,
        metric=spec.metric,
        direction=spec.direction,
        threshold=spec.threshold,
        threshold_mode=spec.threshold_mode,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        improved=improved,
        status=status,
        artifacts=list(spec.artifacts),
    )


def default_registry() -> HarnessRegistry:
    registry = HarnessRegistry()
    registry.register("tiny_metric", tiny_metric_harness)
    return registry


def metric_value(command: str) -> float:
    """Parse deterministic fixture commands such as `metric:tokens_per_second=55`."""
    prefix = "metric:"
    if not command.startswith(prefix):
        raise ValueError(f"Command did not emit fixture metric: {command}")
    _, _, value = command.partition("=")
    if not value:
        raise ValueError(f"Fixture metric command missing value: {command}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Fixture metric value is not numeric: {value}") from exc


def parse_threshold(raw: Any) -> tuple[str, float]:
    if isinstance(raw, (int, float)):
        return "absolute_delta", float(raw)
    text = str(raw).strip()
    if text.startswith("delta>="):
        return "absolute_delta", float(text.removeprefix("delta>=").strip())
    ratio_match = re.fullmatch(
        r"candidate\s*(?:>=|<=)\s*baseline\s*\*\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if ratio_match:
        return "relative_ratio", float(ratio_match.group(1))
    try:
        return "absolute_delta", float(text)
    except ValueError as exc:
        raise ValueError(f"Threshold must be numeric, delta>=N, or candidate >= baseline * R, got: {raw}") from exc
