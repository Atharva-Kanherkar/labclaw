"""Evidence critic with statistical and reproducibility gates for LabClaw."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from labclaw.eval_harness import ExperimentSpec, MetricResult

VALID_VERDICTS = frozenset({"reproduced", "refuted", "inconclusive", "rerun_needed"})
DEFAULT_MIN_SAMPLE_SIZE = 10
DEFAULT_MIN_CONFIDENCE_REPORTABLE = 0.9
FLAKY_FAILURE_PATTERNS = (
    re.compile(r"timed out", re.IGNORECASE),
    re.compile(r"flaky", re.IGNORECASE),
    re.compile(r"intermittent", re.IGNORECASE),
    re.compile(r"connection reset", re.IGNORECASE),
)


@dataclass(frozen=True)
class ReproducibilityContext:
    random_seed: int | None = None
    sample_size: int | None = None
    run_count: int = 1
    flaky: bool = False
    required_artifacts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceInput:
    spec: ExperimentSpec
    metric_result: MetricResult | None = None
    run_status: str | None = None
    command_log: list[dict[str, Any]] | None = None
    artifact_paths: dict[str, str] | None = None
    failure_reason: str | None = None
    reproducibility: ReproducibilityContext | None = None


@dataclass(frozen=True)
class CriticVerdict:
    verdict: str
    confidence: float
    metric_delta: dict[str, Any]
    blocking_objections: list[str]
    reportable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceCritic:
    """Evaluate normalized harness outputs before LabClaw reports a claim."""

    def __init__(
        self,
        *,
        min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
        min_confidence_reportable: float = DEFAULT_MIN_CONFIDENCE_REPORTABLE,
        require_command_log: bool = False,
        require_artifacts: bool = True,
    ) -> None:
        self.min_sample_size = min_sample_size
        self.min_confidence_reportable = min_confidence_reportable
        self.require_command_log = require_command_log
        self.require_artifacts = require_artifacts

    def evaluate(self, evidence: EvidenceInput) -> CriticVerdict:
        objections: list[str] = []
        confidence = 1.0
        metric_delta = build_metric_delta(evidence)
        repro = evidence.reproducibility or ReproducibilityContext()

        if evidence.metric_result is None:
            failure_reason = evidence.failure_reason or ""
            if evidence.run_status == "failed" and (is_flaky_failure(failure_reason) or repro.flaky):
                objections.append(failure_reason or "flaky experiment run")
                return self._finalize(
                    "rerun_needed",
                    confidence=0.3,
                    metric_delta=metric_delta,
                    objections=objections,
                )
            objections.append("missing normalized metric result")
            return self._finalize("inconclusive", confidence=0.0, metric_delta=metric_delta, objections=objections)

        metric = evidence.metric_result
        if metric.metric != evidence.spec.metric:
            objections.append(
                f"metric name mismatch: expected {evidence.spec.metric}, got {metric.metric}"
            )

        if metric.baseline is None:
            objections.append("missing baseline metric")
        if metric.candidate is None:
            objections.append("missing candidate metric")

        if metric.status == "failed":
            failure_reason = metric.failure_reason or evidence.failure_reason or "metric run failed"
            if is_flaky_failure(failure_reason) or repro.flaky:
                objections.append(failure_reason)
                return self._finalize(
                    "rerun_needed",
                    confidence=0.3,
                    metric_delta=metric_delta,
                    objections=objections,
                )
            objections.append(failure_reason)
            return self._finalize("inconclusive", confidence=0.2, metric_delta=metric_delta, objections=objections)

        if evidence.run_status == "failed":
            failure_reason = evidence.failure_reason or "experiment run failed"
            if is_flaky_failure(failure_reason) or repro.flaky:
                objections.append(failure_reason)
                return self._finalize(
                    "rerun_needed",
                    confidence=0.3,
                    metric_delta=metric_delta,
                    objections=objections,
                )
            objections.append(failure_reason)
            return self._finalize("inconclusive", confidence=0.2, metric_delta=metric_delta, objections=objections)

        if self.require_command_log and not evidence.command_log:
            objections.append("missing command log")
        elif not evidence.command_log:
            confidence -= 0.1

        missing_artifacts = missing_required_artifacts(evidence, repro)
        if missing_artifacts:
            if self.require_artifacts:
                for artifact in missing_artifacts:
                    objections.append(f"missing required artifact: {artifact}")
            else:
                confidence -= 0.2

        if repro.random_seed is None:
            confidence -= 0.15
        if repro.sample_size is not None and repro.sample_size < self.min_sample_size:
            confidence -= 0.25
        if repro.run_count < 2 and repro.flaky:
            objections.append("flaky run requires rerun")
            return self._finalize(
                "rerun_needed",
                confidence=0.3,
                metric_delta=metric_delta,
                objections=objections,
            )

        if objections:
            return self._finalize(
                "inconclusive",
                confidence=confidence,
                metric_delta=metric_delta,
                objections=objections,
            )

        if metric.improved:
            return self._finalize(
                "reproduced",
                confidence=confidence,
                metric_delta=metric_delta,
                objections=objections,
            )

        if metric.status == "worse":
            objections.append("candidate metric is worse than baseline")
            return self._finalize(
                "refuted",
                confidence=confidence,
                metric_delta=metric_delta,
                objections=objections,
            )

        objections.append("metric delta did not clear threshold")
        return self._finalize(
            "inconclusive",
            confidence=confidence,
            metric_delta=metric_delta,
            objections=objections,
        )

    def _finalize(
        self,
        verdict: str,
        *,
        confidence: float,
        metric_delta: dict[str, Any],
        objections: list[str],
    ) -> CriticVerdict:
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"Unknown critic verdict: {verdict}")
        bounded_confidence = max(0.0, min(1.0, round(confidence, 3)))
        reportable = (
            verdict == "reproduced"
            and not objections
            and bounded_confidence >= self.min_confidence_reportable
        )
        return CriticVerdict(
            verdict=verdict,
            confidence=bounded_confidence,
            metric_delta=metric_delta,
            blocking_objections=list(objections),
            reportable=reportable,
        )


def build_metric_delta(evidence: EvidenceInput) -> dict[str, Any]:
    metric = evidence.metric_result
    if metric is None:
        return {
            "metric": evidence.spec.metric,
            "baseline": None,
            "candidate": None,
            "delta": None,
            "direction": evidence.spec.direction,
            "threshold": evidence.spec.threshold,
            "threshold_mode": evidence.spec.threshold_mode,
            "improved": False,
            "status": "missing",
        }
    return {
        "metric": metric.metric,
        "baseline": metric.baseline,
        "candidate": metric.candidate,
        "delta": metric.delta,
        "direction": metric.direction,
        "threshold": metric.threshold,
        "threshold_mode": metric.threshold_mode,
        "improved": metric.improved,
        "status": metric.status,
    }


def missing_required_artifacts(
    evidence: EvidenceInput,
    reproducibility: ReproducibilityContext,
) -> list[str]:
    required: list[str] = []
    seen: set[str] = set()
    for path in (
        *evidence.spec.artifacts,
        *(evidence.metric_result.artifacts if evidence.metric_result is not None else ()),
        *reproducibility.required_artifacts,
    ):
        if path not in seen:
            seen.add(path)
            required.append(path)
    if not required:
        return []
    available = set((evidence.artifact_paths or {}).keys())
    return [path for path in required if path not in available]


def is_flaky_failure(reason: str) -> bool:
    return any(pattern.search(reason) for pattern in FLAKY_FAILURE_PATTERNS)


def evidence_from_e2b_result(
    spec: ExperimentSpec,
    *,
    metric_result: MetricResult | None,
    run_status: str,
    commands: list[dict[str, Any]] | None = None,
    artifacts: dict[str, str] | None = None,
    failure_reason: str | None = None,
    reproducibility: ReproducibilityContext | None = None,
) -> EvidenceInput:
    """Build critic input from normalized E2B runner output."""
    return EvidenceInput(
        spec=spec,
        metric_result=metric_result,
        run_status=run_status,
        command_log=commands,
        artifact_paths=artifacts,
        failure_reason=failure_reason,
        reproducibility=reproducibility,
    )
