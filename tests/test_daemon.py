import json
import subprocess
import sys
from pathlib import Path

import pytest

from labclaw.daemon import HeartbeatDaemon
from labclaw.ledger import DEFAULT_STAGES, SCHEMA_VERSION, JsonlRunLedger


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_ledger_creates_heartbeat(tmp_path: Path) -> None:
    ledger = JsonlRunLedger(tmp_path / "ledger.jsonl")

    heartbeat = ledger.create_heartbeat(mission="watch kernels", stages=["scout", "read"])
    stages = ledger.list_stages(heartbeat.run_id)

    assert heartbeat.status == "running"
    assert heartbeat.mission == "watch kernels"
    assert heartbeat.created_at
    assert heartbeat.updated_at
    assert [stage.stage for stage in stages] == ["scout", "read"]
    assert [stage.status for stage in stages] == ["pending", "pending"]


def test_run_ledger_records_stage_failure(tmp_path: Path) -> None:
    ledger = JsonlRunLedger(tmp_path / "ledger.jsonl")
    heartbeat = ledger.create_heartbeat(stages=["scout"])

    ledger.record_stage(heartbeat.run_id, "scout", "running")
    failed_stage = ledger.record_stage(heartbeat.run_id, "scout", "failed", error="network down")
    failed_heartbeat = ledger.update_heartbeat(heartbeat.run_id, "failed")

    assert failed_stage.status == "failed"
    assert failed_stage.error == "network down"
    assert failed_stage.attempt == 1
    assert failed_heartbeat.status == "failed"


def test_run_ledger_writes_schema_version_and_tolerates_old_rows(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    old_heartbeat = {
        "record_type": "heartbeat",
        "run_id": "run-old",
        "mission": "old mission",
        "status": "running",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "future_field": "ignored",
    }
    ledger_path.write_text(json.dumps(old_heartbeat) + "\n", encoding="utf-8")
    ledger = JsonlRunLedger(ledger_path)

    heartbeat = ledger.get_heartbeat("run-old")
    new_heartbeat = ledger.create_heartbeat()
    records = read_jsonl(ledger_path)

    assert heartbeat is not None
    assert heartbeat.schema_version == SCHEMA_VERSION
    assert records[-1]["record_type"] == "stage"
    assert records[-1]["schema_version"] == SCHEMA_VERSION
    assert ledger.get_heartbeat(new_heartbeat.run_id).schema_version == SCHEMA_VERSION


def test_run_ledger_rejects_malformed_jsonl(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text("{not json}\n", encoding="utf-8")
    ledger = JsonlRunLedger(ledger_path)

    with pytest.raises(ValueError, match="Malformed ledger JSON"):
        ledger.get_heartbeat("missing")


def test_heartbeat_daemon_successful_once_run(tmp_path: Path) -> None:
    ledger = JsonlRunLedger(tmp_path / "ledger.jsonl")
    daemon = HeartbeatDaemon(ledger)

    run_id = daemon.run_once(mission="watch claims")

    heartbeat = ledger.get_heartbeat(run_id)
    stages = ledger.list_stages(run_id)
    assert heartbeat is not None
    assert heartbeat.status == "succeeded"
    assert [stage.stage for stage in stages] == DEFAULT_STAGES
    assert all(stage.status == "succeeded" for stage in stages)
    assert all(stage.attempt == 1 for stage in stages)


def test_heartbeat_daemon_rejects_unknown_resume_id(tmp_path: Path) -> None:
    ledger = JsonlRunLedger(tmp_path / "ledger.jsonl")
    daemon = HeartbeatDaemon(ledger)

    with pytest.raises(ValueError, match="unknown heartbeat run"):
        daemon.run_once(resume="missing")


def test_heartbeat_daemon_resumes_failed_run(tmp_path: Path) -> None:
    ledger = JsonlRunLedger(tmp_path / "ledger.jsonl")
    calls = {"read": 0}

    def flaky_read(run_id: str) -> None:
        calls["read"] += 1
        if calls["read"] == 1:
            raise RuntimeError("reader unavailable")

    daemon = HeartbeatDaemon(ledger, handlers={"read": flaky_read})

    with pytest.raises(RuntimeError, match="reader unavailable"):
        daemon.run_once()

    run_id = next(record["run_id"] for record in read_jsonl(ledger.path) if record["record_type"] == "heartbeat")
    assert ledger.get_heartbeat(run_id).status == "failed"
    assert {stage.stage: stage.status for stage in ledger.list_stages(run_id)}["read"] == "failed"

    resumed_id = daemon.run_once(resume=run_id)

    stages = {stage.stage: stage for stage in ledger.list_stages(run_id)}
    assert resumed_id == run_id
    assert ledger.get_heartbeat(run_id).status == "succeeded"
    assert stages["read"].status == "succeeded"
    assert stages["read"].attempt == 2
    assert stages["scout"].attempt == 1
    assert stages["cluster"].attempt == 1


def test_daemon_cli_once_writes_jsonl(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "labclaw.daemon",
            "--once",
            "--ledger",
            str(ledger_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    output = json.loads(result.stdout)
    records = read_jsonl(ledger_path)
    assert output["status"] == "succeeded"
    assert output["run_id"]
    assert any(record["record_type"] == "heartbeat" for record in records)
    assert any(record["record_type"] == "stage" for record in records)


def test_daemon_cli_accepts_daemon_subcommand(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "labclaw.daemon",
            "daemon",
            "--once",
            "--ledger",
            str(ledger_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    output = json.loads(result.stdout)
    assert output["status"] == "succeeded"
    assert output["run_id"]


def test_daemon_cli_resumes_run(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = JsonlRunLedger(ledger_path)
    heartbeat = ledger.create_heartbeat(stages=DEFAULT_STAGES)
    ledger.record_stage(heartbeat.run_id, "scout", "running")
    ledger.record_stage(heartbeat.run_id, "scout", "failed", error="temporary outage")
    ledger.update_heartbeat(heartbeat.run_id, "failed")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "labclaw.daemon",
            "--once",
            "--resume",
            heartbeat.run_id,
            "--ledger",
            str(ledger_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    output = json.loads(result.stdout)
    assert output == {"run_id": heartbeat.run_id, "status": "succeeded"}
    assert ledger.get_heartbeat(heartbeat.run_id).status == "succeeded"
    assert ledger.list_stages(heartbeat.run_id)[0].attempt == 2


def test_daemon_cli_warns_when_resume_mission_is_ignored(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = JsonlRunLedger(ledger_path)
    heartbeat = ledger.create_heartbeat(mission="original mission", stages=["scout"])
    ledger.record_stage(heartbeat.run_id, "scout", "failed", error="temporary outage")
    ledger.update_heartbeat(heartbeat.run_id, "failed")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "labclaw.daemon",
            "daemon",
            "--once",
            "--resume",
            heartbeat.run_id,
            "--mission",
            "new mission",
            "--ledger",
            str(ledger_path),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "ignored when resuming" in result.stderr
    assert json.loads(result.stdout)["status"] == "succeeded"
    assert ledger.get_heartbeat(heartbeat.run_id).mission == "original mission"
