"""Integrated LabClaw heartbeat pipeline for demo and API use."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from labclaw.clustering import ClusterStore
from labclaw.eval_harness import ExperimentSpec, default_registry
from labclaw.evidence_critic import EvidenceCritic, EvidenceInput, ReproducibilityContext
from labclaw.ledger import DEFAULT_MISSION
from labclaw.multimodal_reader import parse_local_fixture, read_source, to_dict as reader_to_dict
from labclaw.report import LabReport, build_report
from labclaw.sources import (
    ARXIV_API,
    GITHUB_SEARCH_API,
    ArxivScout,
    GitHubScout,
    MappingFetcher,
    SeenStore,
    SourceRecord,
    run_scouts,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SAMPLE_SOURCE = Path(__file__).resolve().parent.parent / "samples" / "tiny-ml-claim.md"


def live_reader_enabled() -> bool:
    return bool(os.environ.get("CEREBRAS_API_KEY")) and os.environ.get("LABCLAW_LIVE_READER", "1") not in {
        "0",
        "false",
        "False",
    }


def live_e2b_enabled() -> bool:
    return bool(os.environ.get("E2B_API_KEY")) and os.environ.get("LABCLAW_LIVE_E2B", "0") not in {
        "0",
        "false",
        "False",
    }


def demo_capabilities() -> dict[str, bool]:
    return {
        "fixture_scouts": True,
        "live_reader": live_reader_enabled(),
        "live_e2b": live_e2b_enabled(),
        "gemini_pi": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    }


@dataclass
class StageSnapshot:
    stage: str
    status: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    run_id: str
    mission: str
    stages: list[StageSnapshot]
    source: dict[str, Any]
    cluster: dict[str, Any]
    claim: dict[str, Any]
    experiment_spec: dict[str, Any]
    metric_result: dict[str, Any]
    critic_verdict: dict[str, Any]
    report: dict[str, Any]
    reportable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mission": self.mission,
            "stages": [stage.to_dict() for stage in self.stages],
            "source": self.source,
            "cluster": self.cluster,
            "claim": self.claim,
            "experiment_spec": self.experiment_spec,
            "metric_result": self.metric_result,
            "critic_verdict": self.critic_verdict,
            "report": self.report,
            "reportable": self.reportable,
        }


class LabPipeline:
    """Runs scout → cluster → read → experiment → eval → report in one heartbeat."""

    def __init__(self, data_dir: Path, *, fixture_mode: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.fixture_mode = fixture_mode
        self.runs_dir = self.data_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, mission: str = DEFAULT_MISSION) -> PipelineResult:
        run_id = f"run-{uuid4().hex[:8]}"
        stages: list[StageSnapshot] = []

        sources, scout_summary = self._stage_scout()
        stages.append(StageSnapshot("scout", "succeeded", scout_summary, {"count": len(sources)}))

        source = self._pick_source(sources)
        cluster_store = ClusterStore(self.data_dir / "clusters.json")
        assignment = cluster_store.assign(source)
        cluster_store.save()
        cluster = cluster_store.clusters[assignment.cluster_id]
        stages.append(
            StageSnapshot(
                "cluster",
                "succeeded",
                f"Assigned to {cluster.topic_name}",
                {"assignment": asdict(assignment), "cluster": cluster.to_dict()},
            )
        )

        reader_result, read_summary = self._stage_read(source)
        claim = self._pick_claim(reader_result)
        stages.append(
            StageSnapshot(
                "read",
                "succeeded",
                read_summary,
                reader_to_dict(reader_result),
            )
        )

        spec = self._build_experiment_spec(claim, assignment.cluster_id, source)
        metric, experiment_summary = self._stage_experiment(spec)
        stages.append(
            StageSnapshot(
                "experiment",
                "succeeded" if metric.status != "failed" else "failed",
                experiment_summary,
                {"spec": spec.to_dict()},
            )
        )
        stages.append(
            StageSnapshot(
                "eval",
                "succeeded" if metric.status != "failed" else "failed",
                f"{spec.metric}: {metric.baseline} → {metric.candidate}",
                metric.to_dict(),
            )
        )

        critic = EvidenceCritic(require_artifacts=False)
        plot_path = self.data_dir / "plot.png"
        plot_path.write_bytes(b"demo-plot")
        verdict = critic.evaluate(
            EvidenceInput(
                spec=spec,
                metric_result=metric,
                run_status="succeeded" if metric.status != "failed" else "failed",
                command_log=[
                    {"command": spec.baseline_command, "exit_code": 0},
                    {"command": spec.candidate_command, "exit_code": 0},
                ],
                artifact_paths={"/workspace/plot.png": str(plot_path)},
                reproducibility=ReproducibilityContext(random_seed=42, sample_size=128),
            )
        )
        report = build_report(
            run_id=run_id,
            cluster_topic=cluster.topic_name,
            source=source.to_dict() if isinstance(source, SourceRecord) else dict(source),
            claim=asdict(claim) if hasattr(claim, "__dataclass_fields__") else dict(claim),
            metric_result=metric.to_dict(),
            critic_verdict=verdict.to_dict(),
        )
        stages.append(
            StageSnapshot(
                "report",
                "succeeded",
                "Reportable" if verdict.reportable else f"Held ({verdict.verdict})",
                {"report": report.to_dict(), "markdown": report.to_markdown()},
            )
        )

        result = PipelineResult(
            run_id=run_id,
            mission=mission,
            stages=stages,
            source=source.to_dict() if isinstance(source, SourceRecord) else dict(source),
            cluster=cluster.to_dict(),
            claim=asdict(claim) if hasattr(claim, "__dataclass_fields__") else dict(claim),
            experiment_spec=spec.to_dict(),
            metric_result=metric.to_dict(),
            critic_verdict=verdict.to_dict(),
            report=report.to_dict(),
            reportable=verdict.reportable,
        )
        payload = result.to_dict()
        payload["capabilities"] = demo_capabilities()
        self._persist_payload(result.run_id, payload)
        return result

    def latest(self) -> dict[str, Any] | None:
        runs = sorted(self.runs_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not runs:
            return None
        return json.loads(runs[0].read_text(encoding="utf-8"))

    def _persist(self, result: PipelineResult) -> None:
        payload = result.to_dict()
        payload["capabilities"] = demo_capabilities()
        self._persist_payload(result.run_id, payload)

    def _persist_payload(self, run_id: str, payload: dict[str, Any]) -> None:
        path = self.runs_dir / f"{run_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _stage_scout(self) -> tuple[list[SourceRecord], str]:
        seen = SeenStore(self.data_dir / "seen.json")
        if self.fixture_mode:
            fetcher = MappingFetcher(
                {
                    ARXIV_API: str(FIXTURES_DIR / "arxiv_atom.xml"),
                    GITHUB_SEARCH_API: str(FIXTURES_DIR / "github_search.json"),
                }
            )
            scouts = [ArxivScout(fetcher, max_results=5), GitHubScout(fetcher, max_results=5, fetch_readme=False)]
            records = run_scouts(scouts, seen_store=seen, figure_store=None)
            seen.save()
            sample = self._sample_source_record()
            records.append(sample)
            return records, f"Loaded {len(records)} fixture source(s)"

        from labclaw.figures import FigureStore
        from labclaw.sources import Fetcher

        fetcher = Fetcher()
        figure_store = FigureStore(self.data_dir / "figures", fetcher)
        scouts = [ArxivScout(fetcher, max_results=5), GitHubScout(fetcher, max_results=5)]
        records = run_scouts(scouts, seen_store=seen, figure_store=figure_store)
        seen.save()
        return records, f"Fetched {len(records)} live source(s)"

    def _sample_source_record(self) -> SourceRecord:
        content = SAMPLE_SOURCE.read_text(encoding="utf-8")
        return SourceRecord(
            source_id="sample:tiny-optimizer",
            kind="paper",
            title="Tiny Optimizer Claim",
            url="https://github.com/example/tiny-optimizer",
            published_at="2026-06-01T00:00:00Z",
            raw_text=content,
            figures=[],
            metadata={"topics": ["inference speed", "cache-aware batching"]},
        )

    def _pick_source(self, sources: list[SourceRecord]) -> SourceRecord:
        for record in sources:
            if record.source_id.startswith("sample:"):
                return record
        return sources[0]

    def _stage_read(self, source: SourceRecord) -> tuple[Any, str]:
        if live_reader_enabled():
            try:
                reader_result = read_source(SAMPLE_SOURCE, use_gemma=True)
                return reader_result, f"Live Gemma/Cerebras reader · {len(reader_result.cards)} claim card(s)"
            except Exception as exc:
                reader_result = parse_local_fixture(
                    self._reader_content(source),
                    source_path=SAMPLE_SOURCE,
                )
                return reader_result, f"Fixture reader fallback ({exc})"

        reader_result = parse_local_fixture(
            self._reader_content(source),
            source_path=SAMPLE_SOURCE if self.fixture_mode else None,
        )
        return reader_result, f"Fixture reader · {len(reader_result.cards)} claim card(s)"

    def _stage_experiment(self, spec: ExperimentSpec):
        if live_e2b_enabled():
            try:
                metric = self._run_live_e2b(spec)
                return metric, f"Live E2B sandbox ({metric.status})"
            except Exception as exc:
                metric = default_registry().run(spec)
                return metric, f"Fixture harness fallback after E2B error ({exc})"

        metric = default_registry().run(spec)
        return metric, f"Fixture harness ({metric.status})"

    def _run_live_e2b(self, spec: ExperimentSpec):
        from labclaw.e2b_runner import (
            DEFAULT_E2B_TEMPLATE,
            E2BExperimentRunner,
            E2BRunRequest,
            E2BSandboxFactory,
            ExperimentFile,
        )

        runner = E2BExperimentRunner(
            E2BSandboxFactory(),
            artifact_root=self.data_dir / "e2b-artifacts",
        )
        request = E2BRunRequest(
            spec=spec,
            files=[
                ExperimentFile(
                    path="/workspace/bench.py",
                    content=(
                        "import json, sys\n"
                        "value = 42 if sys.argv[1] == 'baseline' else 55\n"
                        "print(json.dumps({'metrics': {'tokens_per_second': value}}))\n"
                    ),
                )
            ],
            template=os.environ.get("E2B_TEMPLATE", DEFAULT_E2B_TEMPLATE),
            timeout_seconds=90,
        )
        result = runner.run(request)
        if result.metric_result is None:
            raise RuntimeError(result.failure_reason or "E2B run produced no metrics")
        return result.metric_result

    def _reader_content(self, source: SourceRecord) -> str:
        if source.source_id.startswith("sample:") or self.fixture_mode:
            return SAMPLE_SOURCE.read_text(encoding="utf-8")
        return source.raw_text or SAMPLE_SOURCE.read_text(encoding="utf-8")

    def _pick_claim(self, reader_result) -> Any:
        for card in reader_result.cards:
            if card.is_testable:
                return card
        if reader_result.cards:
            return reader_result.cards[0]
        raise ValueError("Reader produced no claim cards.")

    def _build_experiment_spec(self, claim, cluster_id: str, source: SourceRecord) -> ExperimentSpec:
        if live_e2b_enabled():
            baseline_command = "python /workspace/bench.py baseline"
            candidate_command = "python /workspace/bench.py candidate"
            harness = "e2b"
        else:
            baseline_command = "metric:tokens_per_second=42"
            candidate_command = "metric:tokens_per_second=55"
            harness = "tiny_metric"
        return ExperimentSpec(
            claim_id=str(getattr(claim, "id", "claim-1")),
            cluster_id=cluster_id,
            harness=harness,
            baseline_command=baseline_command,
            candidate_command=candidate_command,
            metric="tokens_per_second",
            direction="higher_is_better",
            threshold=5.0,
            source_id=source.source_id,
            artifacts=["/workspace/plot.png"],
            metadata={"demo": True, "claim": getattr(claim, "main_claim", "")},
        )


def build_stage_handlers(pipeline: LabPipeline, *, mission: str = DEFAULT_MISSION):
    """Return daemon stage handlers that execute the integrated pipeline once."""

    state: dict[str, Any] = {"result": None}

    def scout_handler(run_id: str) -> None:
        state["pending"] = pipeline

    def cluster_handler(run_id: str) -> None:
        return None

    def read_handler(run_id: str) -> None:
        return None

    def experiment_handler(run_id: str) -> None:
        return None

    def eval_handler(run_id: str) -> None:
        return None

    def report_handler(run_id: str) -> None:
        if state.get("result") is None:
            state["result"] = pipeline.run(mission=mission)
        return None

    # Run the full pipeline on first stage; later stages are ledger markers only.
    def integrated_handler(run_id: str) -> None:
        if state.get("result") is None:
            state["result"] = pipeline.run(mission=mission)

    return {
        "scout": integrated_handler,
        "cluster": noop_pass,
        "read": noop_pass,
        "experiment": noop_pass,
        "eval": noop_pass,
        "report": noop_pass,
    }


def noop_pass(_run_id: str) -> None:
    return None
