"""E2B-backed experiment runner for bounded LabClaw ExperimentSpecs."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from labclaw.eval_harness import ExperimentSpec, MetricResult

# "base" works in any E2B account out of the box. Pin a custom ML template
# (preinstalled deps) here once it is built; override per-run via E2B_TEMPLATE.
DEFAULT_E2B_TEMPLATE = "base"
FORBIDDEN_SHELL_TOKENS = ("&&", "||", ";", "|", "`", "$(", ">", "<")


class SandboxTimeout(TimeoutError):
    """Raised by sandbox adapters when a command exceeds its timeout."""


@dataclass(frozen=True)
class ExperimentFile:
    path: str
    content: str


@dataclass(frozen=True)
class SandboxCommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class E2BRunRequest:
    spec: ExperimentSpec
    files: list[ExperimentFile] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)
    template: str = DEFAULT_E2B_TEMPLATE
    timeout_seconds: int = 120


@dataclass(frozen=True)
class E2BRunResult:
    claim_id: str
    cluster_id: str
    status: str
    metric_result: MetricResult | None
    commands: list[SandboxCommandResult]
    artifacts: dict[str, str]
    environment: dict[str, Any]
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "cluster_id": self.cluster_id,
            "status": self.status,
            "metric_result": self.metric_result.to_dict() if self.metric_result else None,
            "commands": [command.to_dict() for command in self.commands],
            "artifacts": dict(self.artifacts),
            "environment": dict(self.environment),
            "failure_reason": self.failure_reason,
        }


class SandboxSession(Protocol):
    def write_file(self, path: str, content: str) -> None:
        ...

    def run(self, command: str, *, timeout_seconds: int) -> SandboxCommandResult:
        ...

    def read_file(self, path: str) -> bytes:
        ...

    def environment(self) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class SandboxFactory(Protocol):
    def create(self, *, template: str) -> SandboxSession:
        ...


class E2BExperimentRunner:
    def __init__(self, sandbox_factory: SandboxFactory, *, artifact_root: Path) -> None:
        self.sandbox_factory = sandbox_factory
        self.artifact_root = artifact_root

    def run(self, request: E2BRunRequest) -> E2BRunResult:
        validate_bounded_spec(request.spec)
        validate_commands([*request.setup_commands, request.spec.baseline_command, request.spec.candidate_command])
        sandbox = self.sandbox_factory.create(template=request.template)
        commands: list[SandboxCommandResult] = []
        local_artifacts: dict[str, str] = {}
        metric_result: MetricResult | None = None
        failure_reason: str | None = None
        status = "failed"

        try:
            for experiment_file in request.files:
                sandbox.write_file(experiment_file.path, experiment_file.content)

            for command in request.setup_commands:
                result = self._run_command(sandbox, command, request.timeout_seconds, commands)
                if result.exit_code != 0:
                    failure_reason = f"Setup command failed: {command}"
                    return self._finish(request, sandbox, commands, local_artifacts, metric_result, status, failure_reason)

            baseline_result = self._run_command(
                sandbox,
                request.spec.baseline_command,
                request.timeout_seconds,
                commands,
            )
            if baseline_result.exit_code != 0:
                failure_reason = f"Baseline command failed: {request.spec.baseline_command}"
                return self._finish(request, sandbox, commands, local_artifacts, metric_result, status, failure_reason)

            candidate_result = self._run_command(
                sandbox,
                request.spec.candidate_command,
                request.timeout_seconds,
                commands,
            )
            if candidate_result.exit_code != 0:
                failure_reason = f"Candidate command failed: {request.spec.candidate_command}"
                return self._finish(request, sandbox, commands, local_artifacts, metric_result, status, failure_reason)

            baseline_value = metric_from_stdout(baseline_result.stdout, request.spec.metric)
            candidate_value = metric_from_stdout(candidate_result.stdout, request.spec.metric)
            metric_result = compare_metric_values(request.spec, baseline_value, candidate_value)
            status = "succeeded"

            local_artifacts = self._download_artifacts(sandbox, request)
            return self._finish(request, sandbox, commands, local_artifacts, metric_result, status, None)
        except (SandboxTimeout, ValueError) as exc:
            failure_reason = str(exc)
            return self._finish(request, sandbox, commands, local_artifacts, metric_result, status, failure_reason)
        finally:
            sandbox.close()

    def _run_command(
        self,
        sandbox: SandboxSession,
        command: str,
        timeout_seconds: int,
        commands: list[SandboxCommandResult],
    ) -> SandboxCommandResult:
        started = time.perf_counter()
        try:
            result = sandbox.run(command, timeout_seconds=timeout_seconds)
        except SandboxTimeout as exc:
            commands.append(
                SandboxCommandResult(
                    command=command,
                    stdout="",
                    stderr=str(exc),
                    exit_code=-1,
                    duration_seconds=time.perf_counter() - started,
                )
            )
            raise
        if result.duration_seconds is None:
            result = replace(result, duration_seconds=time.perf_counter() - started)
        commands.append(result)
        return result

    def _download_artifacts(self, sandbox: SandboxSession, request: E2BRunRequest) -> dict[str, str]:
        run_dir = self.artifact_root / request.spec.claim_id
        run_dir.mkdir(parents=True, exist_ok=True)
        local_paths: dict[str, str] = {}
        for sandbox_path in request.artifact_paths:
            local_path = run_dir / Path(sandbox_path).name
            local_path.write_bytes(sandbox.read_file(sandbox_path))
            local_paths[sandbox_path] = str(local_path)
        return local_paths

    def _finish(
        self,
        request: E2BRunRequest,
        sandbox: SandboxSession,
        commands: list[SandboxCommandResult],
        artifacts: dict[str, str],
        metric_result: MetricResult | None,
        status: str,
        failure_reason: str | None,
    ) -> E2BRunResult:
        result = E2BRunResult(
            claim_id=request.spec.claim_id,
            cluster_id=request.spec.cluster_id,
            status=status,
            metric_result=metric_result,
            commands=list(commands),
            artifacts=dict(artifacts),
            environment=sandbox.environment(),
            failure_reason=failure_reason,
        )
        self._write_run_artifacts(request, result)
        return result

    def _write_run_artifacts(self, request: E2BRunRequest, result: E2BRunResult) -> None:
        run_dir = self.artifact_root / request.spec.claim_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "command-log.json").write_text(
            json.dumps([command.to_dict() for command in result.commands], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(result.metric_result.to_dict() if result.metric_result else None, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "environment.json").write_text(
            json.dumps(result.environment, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class E2BSandboxFactory:
    """Optional live E2B adapter. Tests use fake sandboxes instead."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("E2B_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set E2B_API_KEY to use the live E2B experiment runner.")
        try:
            from e2b import Sandbox
        except ImportError as exc:
            raise RuntimeError("Install the E2B Python SDK to use the live E2B experiment runner.") from exc
        self._sandbox_class = Sandbox

    def create(self, *, template: str) -> SandboxSession:
        return E2BSandboxSession(self._sandbox_class.create(template=template, api_key=self.api_key))


class E2BSandboxSession:
    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox

    def write_file(self, path: str, content: str) -> None:
        self.sandbox.files.write(path, content)

    def run(self, command: str, *, timeout_seconds: int) -> SandboxCommandResult:
        from e2b import CommandExitException, TimeoutException

        try:
            result = self.sandbox.commands.run(command, timeout=timeout_seconds)
        except CommandExitException as exc:
            # E2B raises on non-zero exit. Surface it as a result so the runner
            # records the failure and stops cleanly instead of crashing.
            return SandboxCommandResult(
                command=command,
                stdout=str(getattr(exc, "stdout", "") or ""),
                stderr=str(getattr(exc, "stderr", "") or ""),
                exit_code=int(getattr(exc, "exit_code", 1) or 1),
            )
        except TimeoutException as exc:
            raise SandboxTimeout(f"Command timed out after {timeout_seconds}s: {command}") from exc
        return SandboxCommandResult(
            command=command,
            stdout=str(getattr(result, "stdout", "")),
            stderr=str(getattr(result, "stderr", "")),
            exit_code=int(getattr(result, "exit_code", 0)),
        )

    def read_file(self, path: str) -> bytes:
        content = self.sandbox.files.read(path)
        return content if isinstance(content, bytes) else str(content).encode("utf-8")

    def environment(self) -> dict[str, Any]:
        return {"provider": "e2b", "template": getattr(self.sandbox, "template", None)}

    def close(self) -> None:
        kill = getattr(self.sandbox, "kill", None)
        if callable(kill):
            kill()


def validate_bounded_spec(spec: ExperimentSpec) -> None:
    if not spec.baseline_command or not spec.candidate_command:
        raise ValueError("ExperimentSpec must include baseline and candidate commands.")
    if not spec.metric:
        raise ValueError("ExperimentSpec must include a metric name.")


def validate_commands(commands: list[str]) -> None:
    for command in commands:
        if "\n" in command:
            raise ValueError(f"Command must be single-line and bounded: {command!r}")
        if any(token in command for token in FORBIDDEN_SHELL_TOKENS):
            raise ValueError(f"Command contains unsupported shell control token: {command!r}")


def metric_from_stdout(stdout: str, metric: str) -> float:
    try:
        payload = json.loads(stdout.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Command stdout did not contain metric JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Metric JSON must be an object.")
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError("Metric JSON metrics field must be an object.")
    if metric not in metrics:
        raise ValueError(f"Metric JSON missing requested metric: {metric}")
    try:
        return float(metrics[metric])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric JSON value for {metric} must be numeric.") from exc


def compare_metric_values(spec: ExperimentSpec, baseline: float, candidate: float) -> MetricResult:
    delta = candidate - baseline
    if spec.direction == "lower_is_better":
        delta = baseline - candidate
    elif spec.direction != "higher_is_better":
        raise ValueError(f"Unknown metric direction: {spec.direction}")

    if spec.threshold_mode == "absolute_delta":
        improved = delta >= spec.threshold
    elif spec.threshold_mode == "relative_ratio":
        if baseline == 0:
            raise ValueError("Relative threshold cannot compare against zero baseline.")
        ratio = candidate / baseline
        improved = ratio <= spec.threshold if spec.direction == "lower_is_better" else ratio >= spec.threshold
    else:
        raise ValueError(f"Unknown threshold mode: {spec.threshold_mode}")

    status = "improved" if improved else "no_change"
    if not improved and delta < 0:
        status = "worse"
    return MetricResult(
        claim_id=spec.claim_id,
        cluster_id=spec.cluster_id,
        harness=spec.harness,
        metric=spec.metric,
        direction=spec.direction,
        threshold=spec.threshold,
        threshold_mode=spec.threshold_mode,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        improved=improved,
        status=status,
        artifacts=list(spec.artifacts),
    )
