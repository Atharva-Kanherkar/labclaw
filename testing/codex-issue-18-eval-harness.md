# codex/issue-18-eval-harness — Test Contract

## Functional Behavior

- `ExperimentSpec` names a harness and carries claim/cluster provenance, explicit baseline and candidate commands, metric name, improvement direction, threshold, and artifact paths.
- A documented transform builds an `ExperimentSpec` from the Gemini PI `ExperimentProposal` shape introduced in #13.
- The transform rejects PI proposals with `should_run: false` and proposals missing baseline/candidate/metric/threshold fields.
- A harness registry can register named harnesses and run an `ExperimentSpec` by harness name.
- Harness output is normalized into metric JSON with baseline, candidate, delta, threshold, status, artifacts, and failure reason.
- At least one tiny harness compares baseline vs candidate values without network or external credentials.

## Unit Tests

- `test_experiment_spec_from_pi_proposal` — PI proposal becomes an `ExperimentSpec` with provenance and commands.
- `test_experiment_spec_rejects_declined_pi_proposal` — `should_run: false` does not become a runnable spec.
- `test_harness_registry_runs_named_harness` — registry dispatches by harness name and returns normalized metrics.
- `test_tiny_metric_harness_detects_improvement` — candidate clears threshold.
- `test_tiny_metric_harness_detects_no_change` — candidate delta below threshold is `no_change`.
- `test_tiny_metric_harness_detects_worse` — candidate worse than baseline is `worse`.
- `test_tiny_metric_harness_reports_failed_run` — missing/failed baseline or candidate returns `failed` with reason.

## Integration / Functional Tests

- Synthetic PI proposal -> `ExperimentSpec` -> harness registry -> normalized metric result works end-to-end in memory.

## Smoke Tests

- `python -m pytest tests/test_eval_harness.py` runs without network, E2B, or provider credentials.

## E2E Tests

N/A — issue #18 defines local eval semantics. Live E2B execution belongs to #17, and report gating belongs to #19/#20.

## Manual / cURL Tests

N/A — no public CLI is required for this issue.
