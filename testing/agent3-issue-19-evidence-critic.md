# agent3/issue-19-evidence-critic - Test Contract

## Functional Behavior

- `EvidenceCritic` evaluates normalized harness output (`MetricResult`) plus optional run context (command log, artifacts, reproducibility metadata).
- Output matches the issue contract: `verdict`, `confidence`, `metric_delta`, `blocking_objections`, and `reportable`.
- Verdicts are one of `reproduced`, `refuted`, `inconclusive`, or `rerun_needed`.
- The critic refuses reportable output without objective evidence (missing metrics, failed runs, missing artifacts/seeds when required).
- Threshold pass with clean reproducibility context yields `reproduced` and `reportable: true`.
- Soft reproducibility gaps (missing seed, tiny sample, optional missing artifacts/command log) reduce confidence without blocking objections; `min_confidence_reportable` gates `reportable`.
- Threshold fail (`no_change`) yields `inconclusive`, not `refuted`; only `worse` candidates are `refuted`.
- `require_artifacts=False` skips artifact blocking objections and applies only a confidence penalty.
- Flaky failures (timeouts, explicit flaky flags) yield `rerun_needed`.
- `evidence_from_e2b_result()` adapts E2B runner output into critic input without coupling to live sandboxes.

## Unit Tests

- `test_critic_reports_reproduced_when_threshold_passes` - improved metric with clean context is reportable.
- `test_critic_refuses_report_without_objective_evidence` - missing metric result is not reportable.
- `test_critic_flags_missing_baseline` - missing baseline metric blocks a report.
- `test_critic_flags_malformed_metric_name_mismatch` - metric name mismatch is flagged.
- `test_critic_marks_threshold_failure_inconclusive` - below-threshold candidate is inconclusive, not refuted.
- `test_critic_allows_reproduced_when_artifacts_not_required` - optional artifact mode applies confidence penalty only.
- `test_min_confidence_reportable_gates_reproduced_verdict` - soft penalties can block reporting.
- `test_critic_refutes_worse_candidate` - worse-than-baseline candidate is refuted.
- `test_critic_requests_rerun_for_flaky_metric_failure` - timeout-style failures request rerun.
- `test_critic_blocks_report_without_required_artifacts` - missing declared artifacts block reporting.
- `test_critic_blocks_report_without_seed` - missing random seed blocks reporting.
- `test_critic_blocks_report_for_tiny_sample` - tiny sample size blocks reporting.
- `test_critic_evaluates_harness_output_end_to_end` - harness registry output flows through critic.
- `test_evidence_from_e2b_result_builds_critic_input` - E2B-shaped input adapter works.
- `test_critic_output_matches_issue_contract_shape` - JSON-shaped output matches #19 schema.

## Integration / Functional Tests

- Synthetic PI proposal -> harness registry -> critic verdict works in memory without network or E2B credentials.

## Smoke Tests

- `python -m pytest tests/test_evidence_critic.py` runs without network, E2B, or provider credentials.

## E2E Tests

N/A - report delivery belongs to #20 once critic output is wired into the notification pipeline.

## Manual / cURL Tests

N/A - no public CLI is required for this issue.
