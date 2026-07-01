"""Heartbeat daemon for the LabClaw research loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from labclaw.ledger import DEFAULT_MISSION, DEFAULT_STAGES, JsonlRunLedger, RunLedger, StageRun
from labclaw.pipeline import LabPipeline, build_stage_handlers

StageHandler = Callable[[str], None]


class HeartbeatDaemon:
    """Runs the ordered LabClaw stages and records every transition."""

    def __init__(
        self,
        ledger: RunLedger,
        *,
        stages: list[str] | None = None,
        handlers: dict[str, StageHandler] | None = None,
    ) -> None:
        self.ledger = ledger
        self.stages = stages or list(DEFAULT_STAGES)
        self.handlers = handlers or {}

    def run_once(self, *, mission: str = DEFAULT_MISSION, resume: str | None = None) -> str:
        if resume:
            heartbeat = self.ledger.get_heartbeat(resume)
            if heartbeat is None:
                raise ValueError(f"Cannot resume unknown heartbeat run: {resume}")
            run_id = resume
            self.ledger.update_heartbeat(run_id, "running")
        else:
            heartbeat = self.ledger.create_heartbeat(mission=mission, stages=self.stages)
            run_id = heartbeat.run_id

        stages = self.ledger.list_stages(run_id)
        start_index = self._start_index(stages)
        if start_index is None:
            final_status = "succeeded" if all(stage.status == "succeeded" for stage in stages) else "failed"
            self.ledger.update_heartbeat(run_id, final_status)
            return run_id

        for stage in self.stages[start_index:]:
            self.ledger.record_stage(run_id, stage, "running")
            try:
                self.handlers.get(stage, noop_stage_handler)(run_id)
            except Exception as exc:
                # TODO(#22): redact provider secrets before real handlers persist errors.
                self.ledger.record_stage(run_id, stage, "failed", error=str(exc))
                self.ledger.update_heartbeat(run_id, "failed")
                raise
            self.ledger.record_stage(run_id, stage, "succeeded")

        self.ledger.update_heartbeat(run_id, "succeeded")
        return run_id

    def _start_index(self, stages: list[StageRun]) -> int | None:
        by_name = {stage.stage: stage for stage in stages}
        for index, stage in enumerate(self.stages):
            status = by_name.get(stage).status if stage in by_name else "pending"
            if status in ("failed", "pending", "running"):
                return index
        return None


def noop_stage_handler(run_id: str) -> None:
    return None


def load_mission(path: Path | None, fallback: str) -> str:
    if path is None:
        return fallback
    return path.read_text(encoding="utf-8").strip() or fallback


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the LabClaw heartbeat daemon.")
    parser.add_argument("--once", action="store_true", help="Run one heartbeat and exit.")
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume a failed or pending heartbeat.")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(".labclaw") / "run-ledger.jsonl",
        help="Path to the local JSONL run ledger.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("labclaw_data"), help="Pipeline state directory.")
    parser.add_argument("--live", action="store_true", help="Use live scouts/readers instead of fixtures.")
    parser.add_argument("--notify", action="store_true", help="Send Telegram ping when a report is reportable.")
    parser.add_argument("--mission", help="Standing ML mission text.")
    parser.add_argument("--mission-file", type=Path, help="Read standing mission text from a file.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args[:1] == ["daemon"]:
        raw_args = raw_args[1:]
    mission_supplied = any(
        arg in ("--mission", "--mission-file")
        or arg.startswith("--mission=")
        or arg.startswith("--mission-file=")
        for arg in raw_args
    )
    args = parser.parse_args(raw_args)

    if not args.once:
        print("Error: only --once mode is implemented for the MVP daemon.", file=sys.stderr)
        raise SystemExit(2)

    ledger = JsonlRunLedger(args.ledger)
    mission = DEFAULT_MISSION
    if args.resume and mission_supplied:
        print("Warning: --mission and --mission-file are ignored when resuming a run.", file=sys.stderr)
    if not args.resume:
        mission = load_mission(args.mission_file, args.mission or DEFAULT_MISSION)

    pipeline = LabPipeline(
        args.data_dir,
        fixture_mode=not args.live,
        notify_telegram=args.notify,
    )
    handlers = build_stage_handlers(pipeline, mission=mission)
    daemon = HeartbeatDaemon(ledger, handlers=handlers)

    try:
        run_id = daemon.run_once(mission=mission, resume=args.resume)
    except Exception as exc:
        print(f"Heartbeat failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    heartbeat = ledger.get_heartbeat(run_id)
    payload = {"run_id": run_id, "status": heartbeat.status if heartbeat else "unknown"}
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"Heartbeat {payload['run_id']} {payload['status']}.")


if __name__ == "__main__":
    main()
