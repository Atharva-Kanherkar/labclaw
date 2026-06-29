import pytest

from labclaw.eval_harness import ExperimentSpec, MetricResult, default_registry, spec_from_pi_proposal
from labclaw.evidence_critic import (
    CriticVerdict,
    EvidenceCritic,
    EvidenceInput,
    ReproducibilityContext,
    evidence_from_e2b_result,
)


def pi_proposal(**overrides):
    payload = {
        "claim_id": "claim-cache-aware",
        "cluster_id": "cluster-speed",
        "should_run": True,
        "goal": "Compare baseline and cache-aware attention throughput.",
        "baseline_command": "metric:tokens_per_second=42",
        "candidate_command": "metric:tokens_per_second=55",
        "metric": "tokens_per_second",
        "threshold": "delta>=5",
        "rationale": "Explicit commands and measurable speed metric.",
    }
    payload.update(overrides)
    return payload


def metric_result(**overrides) -> MetricResult:
    payload = {
        "claim_id": "claim-cache-aware",
        "cluster_id": "cluster-speed",
        "harness": "tiny_metric",
        "metric": "tokens_per_second",
        "direction": "higher_is_better",
        "threshold": 5.0,
        "threshold_mode": "absolute_delta",
        "baseline": 42.0,
        "candidate": 55.0,
        "delta": 13.0,
        "improved": True,
        "status": "improved",
        "artifacts": ["/workspace/plot.png"],
        "failure_reason": None,
    }
    payload.update(overrides)
    return MetricResult(**payload)


def evidence(**overrides) -> EvidenceInput:
    spec = spec_from_pi_proposal(pi_proposal(), source_id="src-attention")
    payload = {
        "spec": spec,
        "metric_result": metric_result(),
        "run_status": "succeeded",
        "command_log": [{"command": "python bench.py --mode baseline", "exit_code": 0}],
        "artifact_paths": {"/workspace/plot.png": "/tmp/plot.png"},
        "reproducibility": ReproducibilityContext(random_seed=42, sample_size=100),
    }
    payload.update(overrides)
    return EvidenceInput(**payload)


def test_critic_reports_reproduced_when_threshold_passes() -> None:
    verdict = EvidenceCritic().evaluate(evidence())

    assert verdict.verdict == "reproduced"
    assert verdict.reportable is True
    assert verdict.confidence == 1.0
    assert verdict.blocking_objections == []
    assert verdict.metric_delta["delta"] == 13.0
    assert verdict.metric_delta["improved"] is True


def test_critic_refuses_report_without_objective_evidence() -> None:
    verdict = EvidenceCritic().evaluate(evidence(metric_result=None))

    assert verdict.verdict == "inconclusive"
    assert verdict.reportable is False
    assert verdict.confidence == 0.0
    assert "missing normalized metric result" in verdict.blocking_objections


def test_critic_flags_missing_baseline() -> None:
    verdict = EvidenceCritic().evaluate(
        evidence(metric_result=metric_result(baseline=None, delta=None, improved=False, status="failed"))
    )

    assert verdict.verdict == "inconclusive"
    assert verdict.reportable is False
    assert "missing baseline metric" in verdict.blocking_objections


def test_critic_flags_malformed_metric_name_mismatch() -> None:
    verdict = EvidenceCritic().evaluate(
        evidence(metric_result=metric_result(metric="latency", improved=False, status="failed"))
    )

    assert verdict.verdict == "inconclusive"
    assert any("metric name mismatch" in objection for objection in verdict.blocking_objections)


def test_critic_refutes_threshold_failure() -> None:
    verdict = EvidenceCritic().evaluate(
        evidence(
            metric_result=metric_result(
                candidate=45.0,
                delta=3.0,
                improved=False,
                status="no_change",
            )
        )
    )

    assert verdict.verdict == "refuted"
    assert verdict.reportable is False
    assert "metric delta did not clear threshold" in verdict.blocking_objections


def test_critic_refutes_worse_candidate() -> None:
    verdict = EvidenceCritic().evaluate(
        evidence(
            metric_result=metric_result(
                candidate=40.0,
                delta=-2.0,
                improved=False,
                status="worse",
            )
        )
    )

    assert verdict.verdict == "refuted"
    assert verdict.reportable is False
    assert "candidate metric is worse than baseline" in verdict.blocking_objections


def test_critic_requests_rerun_for_flaky_timeout() -> None:
    verdict = EvidenceCritic().evaluate(
        evidence(
            metric_result=None,
            run_status="failed",
            failure_reason="Command timed out after 30s: python bench.py --mode baseline",
        )
    )

    assert verdict.verdict == "rerun_needed"
    assert verdict.reportable is False
    assert verdict.confidence <= 0.3


def test_critic_requests_rerun_for_flaky_metric_failure() -> None:
    verdict = EvidenceCritic().evaluate(
        evidence(
            metric_result=metric_result(
                baseline=None,
                candidate=None,
                delta=None,
                improved=False,
                status="failed",
                failure_reason="Command timed out after 30s",
            )
        )
    )

    assert verdict.verdict == "rerun_needed"
    assert verdict.reportable is False
    assert verdict.confidence <= 0.3


def test_critic_blocks_report_without_required_artifacts() -> None:
    verdict = EvidenceCritic(require_artifacts=True).evaluate(evidence(artifact_paths={}))

    assert verdict.verdict == "inconclusive"
    assert verdict.reportable is False
    assert any("missing required artifact" in objection for objection in verdict.blocking_objections)


def test_critic_blocks_report_without_seed() -> None:
    verdict = EvidenceCritic().evaluate(
        evidence(reproducibility=ReproducibilityContext(random_seed=None, sample_size=100))
    )

    assert verdict.verdict == "inconclusive"
    assert verdict.reportable is False
    assert "missing random seed" in verdict.blocking_objections


def test_critic_blocks_report_for_tiny_sample() -> None:
    verdict = EvidenceCritic(min_sample_size=50).evaluate(
        evidence(reproducibility=ReproducibilityContext(random_seed=7, sample_size=5))
    )

    assert verdict.verdict == "inconclusive"
    assert verdict.reportable is False
    assert any("sample size below minimum" in objection for objection in verdict.blocking_objections)


def test_critic_evaluates_harness_output_end_to_end() -> None:
    spec = spec_from_pi_proposal(pi_proposal())
    metric = default_registry().run(spec)
    verdict = EvidenceCritic(require_artifacts=False).evaluate(
        EvidenceInput(
            spec=spec,
            metric_result=metric,
            run_status="succeeded",
            reproducibility=ReproducibilityContext(random_seed=1, sample_size=128),
        )
    )

    assert metric.status == "improved"
    assert verdict.verdict == "reproduced"
    assert verdict.reportable is True


def test_evidence_from_e2b_result_builds_critic_input() -> None:
    spec = spec_from_pi_proposal(pi_proposal())
    metric = default_registry().run(spec)
    evidence_input = evidence_from_e2b_result(
        spec,
        metric_result=metric,
        run_status="succeeded",
        commands=[{"command": spec.baseline_command, "exit_code": 0}],
        artifacts={"/workspace/plot.png": "/tmp/plot.png"},
        reproducibility=ReproducibilityContext(random_seed=99, sample_size=64),
    )

    verdict = EvidenceCritic().evaluate(evidence_input)

    assert isinstance(verdict, CriticVerdict)
    assert verdict.verdict == "reproduced"
    assert verdict.to_dict()["metric_delta"]["candidate"] == 55.0


def test_critic_output_matches_issue_contract_shape() -> None:
    verdict = EvidenceCritic().evaluate(evidence()).to_dict()

    assert set(verdict) == {"verdict", "confidence", "metric_delta", "blocking_objections", "reportable"}
    assert verdict["verdict"] in {"reproduced", "refuted", "inconclusive", "rerun_needed"}
    assert isinstance(verdict["confidence"], float)
    assert isinstance(verdict["metric_delta"], dict)
    assert isinstance(verdict["blocking_objections"], list)
    assert isinstance(verdict["reportable"], bool)
