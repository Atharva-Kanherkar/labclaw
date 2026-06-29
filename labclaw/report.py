"""Report pipeline for LabClaw demo and Telegram-ready output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LabReport:
    run_id: str
    cluster_topic: str
    source_title: str
    source_url: str
    claim: str
    baseline_metric: float | None
    candidate_metric: float | None
    metric_name: str
    metric_delta: float | None
    verdict: str
    confidence: float
    reportable: bool
    why_it_matters: str
    recommended_next_action: str
    artifact_paths: list[str] = field(default_factory=list)
    blocking_objections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        baseline = "n/a" if self.baseline_metric is None else f"{self.baseline_metric:g}"
        candidate = "n/a" if self.candidate_metric is None else f"{self.candidate_metric:g}"
        delta = "n/a" if self.metric_delta is None else f"{self.metric_delta:+g}"
        lines = [
            f"# LabClaw report · {self.run_id}",
            "",
            f"**Cluster:** {self.cluster_topic}",
            f"**Source:** [{self.source_title}]({self.source_url})",
            "",
            "## Claim tested",
            self.claim,
            "",
            "## Metrics",
            f"- Metric: `{self.metric_name}`",
            f"- Baseline: {baseline}",
            f"- Candidate: {candidate}",
            f"- Delta: {delta}",
            "",
            "## Verdict",
            f"- Result: **{self.verdict}**",
            f"- Confidence: {self.confidence:.2f}",
            f"- Reportable: {self.reportable}",
            "",
            "## Why it matters",
            self.why_it_matters,
            "",
            "## Next action",
            self.recommended_next_action,
        ]
        if self.blocking_objections:
            lines.extend(["", "## Blocking objections", *[f"- {item}" for item in self.blocking_objections]])
        return "\n".join(lines)

    def telegram_ping(self) -> str:
        if not self.reportable:
            return ""
        baseline = "n/a" if self.baseline_metric is None else f"{self.baseline_metric:g}"
        candidate = "n/a" if self.candidate_metric is None else f"{self.candidate_metric:g}"
        return (
            f"LabClaw · {self.verdict}\n"
            f"{self.cluster_topic}\n"
            f"{self.claim[:180]}\n"
            f"{self.metric_name}: {baseline} → {candidate}\n"
            f"Δ {self.metric_delta:+g} · confidence {self.confidence:.2f}"
        )


def build_report(
    *,
    run_id: str,
    cluster_topic: str,
    source: Mapping[str, Any],
    claim: Mapping[str, Any],
    metric_result: Mapping[str, Any] | None,
    critic_verdict: Mapping[str, Any],
) -> LabReport:
    metric_delta = critic_verdict.get("metric_delta") or {}
    verdict = str(critic_verdict.get("verdict", "inconclusive"))
    reportable = bool(critic_verdict.get("reportable", False))
    claim_text = str(claim.get("main_claim") or claim.get("claim") or "Untitled claim")
    metric_name = str(metric_delta.get("metric") or (metric_result or {}).get("metric") or "metric")

    if verdict == "reproduced" and reportable:
        why = "Measured baseline vs candidate cleared the threshold with reproducibility checks."
        next_action = "Promote the candidate approach inside this cluster and schedule a follow-up experiment."
    elif verdict == "refuted":
        why = "The candidate underperformed or contradicted the claimed improvement."
        next_action = "Archive the claim in cluster memory and scout for a stronger candidate."
    elif verdict == "rerun_needed":
        why = "The run was flaky or incomplete; no honest report can be sent yet."
        next_action = "Rerun with fixed seeds, larger sample, and stable artifacts."
    else:
        why = "Evidence exists but is not strong enough to wake the user."
        next_action = "Keep watching the cluster; do not notify until reportable evidence exists."

    artifacts = list(claim.get("artifacts") or [])
    if metric_result:
        artifacts.extend(metric_result.get("artifacts") or [])

    return LabReport(
        run_id=run_id,
        cluster_topic=cluster_topic,
        source_title=str(source.get("title") or "Unknown source"),
        source_url=str(source.get("url") or ""),
        claim=claim_text,
        baseline_metric=metric_delta.get("baseline"),
        candidate_metric=metric_delta.get("candidate"),
        metric_name=metric_name,
        metric_delta=metric_delta.get("delta"),
        verdict=verdict,
        confidence=float(critic_verdict.get("confidence") or 0.0),
        reportable=reportable,
        why_it_matters=why,
        recommended_next_action=next_action,
        artifact_paths=sorted(set(str(path) for path in artifacts if path)),
        blocking_objections=list(critic_verdict.get("blocking_objections") or []),
    )
