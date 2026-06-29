import pytest

from labclaw.pipeline import LabPipeline


def test_demo_pipeline_runs_end_to_end(tmp_path) -> None:
    pipeline = LabPipeline(tmp_path / "data", fixture_mode=True)
    result = pipeline.run(mission="Hackathon demo heartbeat")

    assert result.run_id.startswith("run-")
    assert len(result.stages) == 6
    assert result.source["source_id"] == "sample:tiny-optimizer"
    assert result.cluster["cluster_id"]
    assert result.claim["main_claim"]
    assert result.metric_result["status"] == "improved"
    assert result.critic_verdict["verdict"] == "reproduced"
    assert result.reportable is True
    assert result.report["reportable"] is True
    assert result.report["why_it_matters"]
    assert "LabClaw report" in result.stages[-1].payload["markdown"]


def test_demo_pipeline_persists_latest(tmp_path) -> None:
    pipeline = LabPipeline(tmp_path / "data", fixture_mode=True)
    result = pipeline.run()
    latest = pipeline.latest()
    assert latest is not None
    assert latest["run_id"] == result.run_id
