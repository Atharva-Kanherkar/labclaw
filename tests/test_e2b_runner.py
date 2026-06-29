import json
import os
from pathlib import Path

import pytest

from labclaw.e2b_runner import (
    DEFAULT_E2B_TEMPLATE,
    E2BExperimentRunner,
    E2BRunRequest,
    E2BSandboxFactory,
    ExperimentFile,
    SandboxCommandResult,
    SandboxTimeout,
)
from labclaw.eval_harness import ExperimentSpec


class FakeSandbox:
    def __init__(self, responses=None, files=None, timeout_on=None):
        self.responses = dict(responses or {})
        self.files = dict(files or {})
        self.timeout_on = timeout_on
        self.writes = []
        self.runs = []
        self.closed = False

    def write_file(self, path: str, content: str) -> None:
        self.writes.append((path, content))
        self.files[path] = content.encode("utf-8")

    def run(self, command: str, *, timeout_seconds: int) -> SandboxCommandResult:
        self.runs.append((command, timeout_seconds))
        if command == self.timeout_on:
            raise SandboxTimeout(f"Command timed out: {command}")
        return self.responses.get(
            command,
            SandboxCommandResult(command=command, stdout="", stderr="unknown command", exit_code=127),
        )

    def read_file(self, path: str) -> bytes:
        return self.files[path]

    def environment(self):
        return {"provider": "fake-e2b", "python": "3.11"}

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self, sandbox: FakeSandbox):
        self.sandbox = sandbox
        self.templates = []

    def create(self, *, template: str) -> FakeSandbox:
        self.templates.append(template)
        return self.sandbox


def command_result(command, metric=None, *, exit_code=0, stderr=""):
    stdout = json.dumps({"metrics": metric}) if metric is not None else ""
    return SandboxCommandResult(command=command, stdout=stdout, stderr=stderr, exit_code=exit_code)


def experiment_spec(**overrides):
    payload = {
        "claim_id": "claim-cache-aware",
        "cluster_id": "cluster-speed",
        "harness": "e2b",
        "baseline_command": "python bench.py --mode baseline",
        "candidate_command": "python bench.py --mode candidate",
        "metric": "tokens_per_second",
        "direction": "higher_is_better",
        "threshold": 5.0,
        "threshold_mode": "absolute_delta",
        "artifacts": ["/workspace/plot.png"],
    }
    payload.update(overrides)
    return ExperimentSpec(**payload)


def successful_sandbox() -> FakeSandbox:
    return FakeSandbox(
        responses={
            "pip install -r requirements.txt": command_result("pip install -r requirements.txt"),
            "python bench.py --mode baseline": command_result(
                "python bench.py --mode baseline",
                {"tokens_per_second": 42},
            ),
            "python bench.py --mode candidate": command_result(
                "python bench.py --mode candidate",
                {"tokens_per_second": 55},
            ),
        },
        files={"/workspace/plot.png": b"fake-png"},
    )


def test_e2b_runner_uploads_files_and_runs_baseline_candidate(tmp_path: Path) -> None:
    sandbox = successful_sandbox()
    runner = E2BExperimentRunner(FakeFactory(sandbox), artifact_root=tmp_path)
    request = E2BRunRequest(
        spec=experiment_spec(),
        files=[ExperimentFile(path="/workspace/bench.py", content="print('bench')")],
        setup_commands=["pip install -r requirements.txt"],
        artifact_paths=["/workspace/plot.png"],
        template="labclaw-test-template",
        timeout_seconds=30,
    )

    result = runner.run(request)

    assert result.status == "succeeded"
    assert result.metric_result.status == "improved"
    assert sandbox.writes == [("/workspace/bench.py", "print('bench')")]
    assert sandbox.runs == [
        ("pip install -r requirements.txt", 30),
        ("python bench.py --mode baseline", 30),
        ("python bench.py --mode candidate", 30),
    ]
    assert sandbox.closed is True


def test_e2b_runner_writes_local_artifacts(tmp_path: Path) -> None:
    sandbox = successful_sandbox()
    runner = E2BExperimentRunner(FakeFactory(sandbox), artifact_root=tmp_path)

    result = runner.run(E2BRunRequest(spec=experiment_spec(), artifact_paths=["/workspace/plot.png"]))

    run_dir = tmp_path / "claim-cache-aware"
    assert (run_dir / "command-log.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "plot.png").read_bytes() == b"fake-png"
    assert result.artifacts == {"/workspace/plot.png": str(run_dir / "plot.png")}
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["candidate"] == 55.0
    assert metrics["delta"] == 13.0


def test_e2b_runner_rejects_unbounded_shell_commands(tmp_path: Path) -> None:
    sandbox = successful_sandbox()
    runner = E2BExperimentRunner(FakeFactory(sandbox), artifact_root=tmp_path)
    spec = experiment_spec(candidate_command="python bench.py --mode candidate && rm -rf /")

    with pytest.raises(ValueError, match="shell control token"):
        runner.run(E2BRunRequest(spec=spec))

    assert sandbox.runs == []


def test_e2b_runner_handles_setup_failure(tmp_path: Path) -> None:
    sandbox = successful_sandbox()
    sandbox.responses["pip install -r requirements.txt"] = command_result(
        "pip install -r requirements.txt",
        exit_code=1,
        stderr="install failed",
    )
    runner = E2BExperimentRunner(FakeFactory(sandbox), artifact_root=tmp_path)

    result = runner.run(E2BRunRequest(spec=experiment_spec(), setup_commands=["pip install -r requirements.txt"]))

    assert result.status == "failed"
    assert result.metric_result is None
    assert result.failure_reason == "Setup command failed: pip install -r requirements.txt"
    assert len(result.commands) == 1
    assert result.commands[0].stderr == "install failed"


def test_e2b_runner_handles_candidate_failure(tmp_path: Path) -> None:
    sandbox = successful_sandbox()
    sandbox.responses["python bench.py --mode candidate"] = command_result(
        "python bench.py --mode candidate",
        exit_code=2,
        stderr="candidate crashed",
    )
    runner = E2BExperimentRunner(FakeFactory(sandbox), artifact_root=tmp_path)

    result = runner.run(E2BRunRequest(spec=experiment_spec()))

    assert result.status == "failed"
    assert result.failure_reason == "Candidate command failed: python bench.py --mode candidate"
    assert result.commands[-1].exit_code == 2
    assert result.commands[-1].stderr == "candidate crashed"


def test_e2b_runner_handles_timeout(tmp_path: Path) -> None:
    sandbox = successful_sandbox()
    sandbox.timeout_on = "python bench.py --mode baseline"
    runner = E2BExperimentRunner(FakeFactory(sandbox), artifact_root=tmp_path)

    result = runner.run(E2BRunRequest(spec=experiment_spec()))

    assert result.status == "failed"
    assert "timed out" in result.failure_reason
    assert result.metric_result is None
    assert result.commands[-1].command == "python bench.py --mode baseline"
    assert result.commands[-1].exit_code == -1


def test_e2b_runner_handles_malformed_metric_json(tmp_path: Path) -> None:
    sandbox = successful_sandbox()
    sandbox.responses["python bench.py --mode baseline"] = SandboxCommandResult(
        command="python bench.py --mode baseline",
        stdout="not-json",
        stderr="",
        exit_code=0,
    )
    runner = E2BExperimentRunner(FakeFactory(sandbox), artifact_root=tmp_path)

    result = runner.run(E2BRunRequest(spec=experiment_spec()))

    assert result.status == "failed"
    assert "metric JSON" in result.failure_reason


def test_e2b_runner_handles_missing_metric(tmp_path: Path) -> None:
    sandbox = successful_sandbox()
    sandbox.responses["python bench.py --mode candidate"] = command_result(
        "python bench.py --mode candidate",
        {"latency": 12},
    )
    runner = E2BExperimentRunner(FakeFactory(sandbox), artifact_root=tmp_path)

    result = runner.run(E2BRunRequest(spec=experiment_spec()))

    assert result.status == "failed"
    assert "missing requested metric" in result.failure_reason


def test_live_e2b_factory_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="E2B_API_KEY"):
        E2BSandboxFactory()


def test_live_e2b_smoke_when_credentials_available(tmp_path: Path) -> None:
    if not os.environ.get("E2B_API_KEY"):
        pytest.skip("Set E2B_API_KEY to run the live E2B smoke test.")
    pytest.importorskip("e2b")

    runner = E2BExperimentRunner(E2BSandboxFactory(), artifact_root=tmp_path)
    request = E2BRunRequest(
        spec=experiment_spec(
            baseline_command="python /workspace/bench.py baseline",
            candidate_command="python /workspace/bench.py candidate",
        ),
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
        timeout_seconds=60,
    )

    result = runner.run(request)

    assert result.status == "succeeded"
    assert result.metric_result.improved is True
