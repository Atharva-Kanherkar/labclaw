"""Local run ledger for LabClaw heartbeats.

The ledger is intentionally append-only JSONL. Callers depend on the
RunLedger interface, so the backing store can later move to SQLite without
changing the daemon.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

DEFAULT_STAGES = ["scout", "cluster", "read", "experiment", "eval", "report"]
DEFAULT_MISSION = "Watch ML/code research and report only measured evidence."
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class HeartbeatRun:
    run_id: str
    mission: str
    status: str
    created_at: str
    updated_at: str
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class StageRun:
    run_id: str
    stage: str
    status: str
    attempt: int
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    schema_version: int = SCHEMA_VERSION


class RunLedger(Protocol):
    def create_heartbeat(
        self, mission: str = DEFAULT_MISSION, stages: list[str] | None = None
    ) -> HeartbeatRun:
        ...

    def get_heartbeat(self, run_id: str) -> HeartbeatRun | None:
        ...

    def list_stages(self, run_id: str) -> list[StageRun]:
        ...

    def record_stage(self, run_id: str, stage: str, status: str, error: str | None = None) -> StageRun:
        ...

    def update_heartbeat(self, run_id: str, status: str) -> HeartbeatRun:
        ...


class JsonlRunLedger:
    """Append-only JSONL ledger with latest-record reconstruction."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def create_heartbeat(
        self, mission: str = DEFAULT_MISSION, stages: list[str] | None = None
    ) -> HeartbeatRun:
        now = utc_now()
        heartbeat = HeartbeatRun(
            run_id=uuid4().hex,
            mission=mission,
            status="running",
            created_at=now,
            updated_at=now,
        )
        self._append("heartbeat", asdict(heartbeat))
        for stage in stages or DEFAULT_STAGES:
            self._append(
                "stage",
                asdict(
                    StageRun(
                        run_id=heartbeat.run_id,
                        stage=stage,
                        status="pending",
                        attempt=0,
                        created_at=now,
                        updated_at=now,
                    )
                ),
            )
        return heartbeat

    def get_heartbeat(self, run_id: str) -> HeartbeatRun | None:
        heartbeats, _ = self._load_latest()
        return heartbeats.get(run_id)

    def list_stages(self, run_id: str) -> list[StageRun]:
        _, stages = self._load_latest()
        return [stage for (stage_run_id, _), stage in stages.items() if stage_run_id == run_id]

    def record_stage(self, run_id: str, stage: str, status: str, error: str | None = None) -> StageRun:
        previous = self._stage(run_id, stage)
        now = utc_now()
        attempt = previous.attempt if previous else 0
        started_at = previous.started_at if previous else None
        if status == "running":
            attempt += 1
            started_at = now
        record = StageRun(
            run_id=run_id,
            stage=stage,
            status=status,
            attempt=attempt,
            created_at=previous.created_at if previous else now,
            updated_at=now,
            started_at=started_at,
            finished_at=now if status in ("succeeded", "failed", "skipped") else None,
            error=error if status == "failed" else None,
        )
        self._append("stage", asdict(record))
        return record

    def update_heartbeat(self, run_id: str, status: str) -> HeartbeatRun:
        previous = self.get_heartbeat(run_id)
        if previous is None:
            raise KeyError(f"Unknown heartbeat run: {run_id}")
        heartbeat = HeartbeatRun(
            run_id=previous.run_id,
            mission=previous.mission,
            status=status,
            created_at=previous.created_at,
            updated_at=utc_now(),
        )
        self._append("heartbeat", asdict(heartbeat))
        return heartbeat

    def _stage(self, run_id: str, stage: str) -> StageRun | None:
        _, stages = self._load_latest()
        return stages.get((run_id, stage))

    def _append(self, record_type: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"record_type": record_type, **payload}, sort_keys=True) + "\n")

    def _load_latest(self) -> tuple[dict[str, HeartbeatRun], dict[tuple[str, str], StageRun]]:
        heartbeats: dict[str, HeartbeatRun] = {}
        stages: dict[tuple[str, str], StageRun] = {}
        if not self.path.exists():
            return heartbeats, stages

        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed ledger JSON on line {line_number}") from exc
                record_type = record.pop("record_type", None)
                if record_type == "heartbeat":
                    heartbeat = dataclass_from_record(HeartbeatRun, record)
                    heartbeats[heartbeat.run_id] = heartbeat
                elif record_type == "stage":
                    stage = dataclass_from_record(StageRun, record)
                    stages[(stage.run_id, stage.stage)] = stage
                else:
                    raise ValueError(f"Unknown ledger record type on line {line_number}: {record_type}")
        return heartbeats, stages


def dataclass_from_record(record_class: type[HeartbeatRun] | type[StageRun], record: dict[str, Any]):
    allowed_fields = {field.name for field in fields(record_class)}
    return record_class(**{key: value for key, value in record.items() if key in allowed_fields})
