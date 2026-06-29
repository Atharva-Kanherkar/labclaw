# codex/issue-17-e2b-runner - Test Contract

## Functional Behavior

- `E2BExperimentRunner` accepts a bounded #18 `ExperimentSpec`; it does not accept arbitrary free-form command lists.
- The runner starts a sandbox from a pinned template through an injectable sandbox factory.
- The runner uploads experiment files before commands execute.
- The runner runs setup commands, then baseline command, then candidate command with timeouts.
- Command stdout/stderr/exit code/duration are captured in order.
- Baseline and candidate commands must emit metric JSON containing the requested metric name.
- The runner compares baseline and candidate with the #18 eval semantics and writes local artifacts:
  command log JSON, metric JSON, environment JSON, and any requested sandbox artifact files.
- Failed setup, failed baseline/candidate, malformed metric JSON, missing metrics, and timeouts are represented cleanly with `status="failed"` and a failure reason.
- A live E2B adapter is optional and requires `E2B_API_KEY`; tests use a fake sandbox and never require credentials.

## Unit Tests

- `test_e2b_runner_uploads_files_and_runs_baseline_candidate` - fake sandbox receives files and ordered commands.
- `test_e2b_runner_writes_local_artifacts` - command log, metric JSON, environment JSON, and downloaded artifact are saved.
- `test_e2b_runner_rejects_unbounded_shell_commands` - shell metacharacters/pipelines are rejected before sandbox execution.
- `test_e2b_runner_handles_setup_failure` - setup failure stops the run and records failure reason.
- `test_e2b_runner_handles_candidate_failure` - failed candidate command records stdout/stderr/exit code and failure reason.
- `test_e2b_runner_handles_timeout` - timeout errors become failed run results.
- `test_e2b_runner_handles_malformed_metric_json` - bad metric output fails cleanly.
- `test_live_e2b_factory_requires_api_key` - live factory construction fails clearly without `E2B_API_KEY`.

## Integration / Functional Tests

- Deterministic toy ML experiment uses fake E2B commands that emit baseline/candidate metric JSON, produces a normalized improved result, and downloads a plot artifact.

## Smoke Tests

- `python -m pytest tests/test_e2b_runner.py` runs without network or live E2B credentials.

## E2E Tests

N/A for automated CI - live E2B smoke requires real `E2B_API_KEY` and is manual.

## Manual / cURL Tests

With credentials and the optional dependency installed:

```bash
export E2B_API_KEY=...
python -m pytest tests/test_e2b_runner.py
```
