import json
from pathlib import Path

from labclaw.daemon import HeartbeatDaemon
from labclaw.ledger import JsonlRunLedger
from labclaw.pipeline import LabPipeline, build_stage_handlers


def test_daemon_runs_wired_pipeline(tmp_path: Path) -> None:
    ledger = JsonlRunLedger(tmp_path / "ledger.jsonl")
    pipeline = LabPipeline(tmp_path / "data", fixture_mode=True)
    handlers = build_stage_handlers(pipeline, mission="daemon wiring test")
    daemon = HeartbeatDaemon(ledger, handlers=handlers)

    run_id = daemon.run_once(mission="daemon wiring test")

    assert ledger.get_heartbeat(run_id).status == "succeeded"
    latest = pipeline.latest()
    assert latest is not None
    assert latest["mission"] == "daemon wiring test"
    assert latest["reportable"] is True
