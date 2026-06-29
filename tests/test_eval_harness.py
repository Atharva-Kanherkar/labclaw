import pytest

from labclaw.eval_harness import (
    ExperimentSpec,
    HarnessRegistry,
    default_registry,
    spec_from_pi_proposal,
    tiny_metric_harness,
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


def test_experiment_spec_from_pi_proposal() -> None:
    spec = spec_from_pi_proposal(pi_proposal(), source_id="src-attention")

    assert spec.claim_id == "claim-cache-aware"
    assert spec.cluster_id == "cluster-speed"
    assert spec.source_id == "src-attention"
    assert spec.harness == "tiny_metric"
    assert spec.baseline_command == "metric:tokens_per_second=42"
    assert spec.candidate_command == "metric:tokens_per_second=55"
    assert spec.metric == "tokens_per_second"
    assert spec.direction == "higher_is_better"
    assert spec.threshold == 5.0
    assert spec.threshold_mode == "absolute_delta"
    assert spec.metadata["pi_rationale"].startswith("Explicit commands")


def test_experiment_spec_rejects_declined_pi_proposal() -> None:
    with pytest.raises(ValueError, match="declined"):
        spec_from_pi_proposal(pi_proposal(should_run=False, baseline_command="", candidate_command=""))


def test_experiment_spec_rejects_missing_pi_command() -> None:
    with pytest.raises(ValueError, match="candidate_command"):
        spec_from_pi_proposal(pi_proposal(candidate_command=""))


def test_harness_registry_runs_named_harness() -> None:
    registry = HarnessRegistry()
    registry.register("tiny_metric", tiny_metric_harness)
    spec = spec_from_pi_proposal(pi_proposal())

    result = registry.run(spec)

    assert result.status == "improved"
    assert result.to_dict()["candidate"] == 55.0


def test_harness_registry_rejects_unknown_harness() -> None:
    registry = HarnessRegistry()
    spec = spec_from_pi_proposal(pi_proposal(), harness="missing")

    with pytest.raises(KeyError, match="Unknown eval harness"):
        registry.run(spec)


def test_tiny_metric_harness_detects_improvement() -> None:
    spec = spec_from_pi_proposal(pi_proposal())

    result = default_registry().run(spec)

    assert result.status == "improved"
    assert result.improved is True
    assert result.baseline == 42.0
    assert result.candidate == 55.0
    assert result.delta == 13.0
    assert result.threshold == 5.0
    assert result.threshold_mode == "absolute_delta"
    assert result.failure_reason is None


def test_tiny_metric_harness_supports_relative_ratio_threshold() -> None:
    spec = spec_from_pi_proposal(pi_proposal(threshold="candidate >= baseline * 1.10"))

    result = default_registry().run(spec)

    assert result.status == "improved"
    assert result.threshold == 1.10
    assert result.threshold_mode == "relative_ratio"
    assert result.delta == 13.0


def test_tiny_metric_harness_detects_no_change() -> None:
    spec = spec_from_pi_proposal(pi_proposal(candidate_command="metric:tokens_per_second=45"))

    result = default_registry().run(spec)

    assert result.status == "no_change"
    assert result.improved is False
    assert result.delta == 3.0


def test_tiny_metric_harness_detects_worse() -> None:
    spec = spec_from_pi_proposal(pi_proposal(candidate_command="metric:tokens_per_second=40"))

    result = default_registry().run(spec)

    assert result.status == "worse"
    assert result.improved is False
    assert result.delta == -2.0


def test_tiny_metric_harness_supports_lower_is_better() -> None:
    spec = spec_from_pi_proposal(
        pi_proposal(
            baseline_command="metric:validation_loss=1.2",
            candidate_command="metric:validation_loss=0.9",
            metric="validation_loss",
            threshold="delta>=0.1",
        ),
        direction="lower_is_better",
    )

    result = default_registry().run(spec)

    assert result.status == "improved"
    assert result.delta == pytest.approx(0.3)


def test_tiny_metric_harness_reports_failed_run() -> None:
    spec = ExperimentSpec(
        claim_id="claim-broken",
        cluster_id="cluster-speed",
        harness="tiny_metric",
        baseline_command="metric:tokens_per_second=42",
        candidate_command="python bench.py --mode candidate",
        metric="tokens_per_second",
        direction="higher_is_better",
        threshold=5.0,
    )

    result = default_registry().run(spec)

    assert result.status == "failed"
    assert result.improved is False
    assert result.baseline is None
    assert result.candidate is None
    assert result.delta is None
    assert "did not emit fixture metric" in result.failure_reason
