# codex/claim-picker — Test Contract

## Functional Behavior

- Select the best claim card from a reader result.
- Prefer claims with runnable code hooks, benchmark numbers, figure evidence, and explicit evidence needs.
- Return `None` when no card has a credible reproduction path.
- Emit an experiment spec with objective success signals.
- Provide a CLI that reads reader JSON and prints the selected experiment spec.

## Unit Tests

- `test_pick_claim_prefers_runnable_measured_visual_claim`
- `test_pick_claim_returns_none_without_reproduction_path`
- `test_build_experiment_spec_uses_first_safe_command_sequence`
- `test_pick_claim_tie_breaks_by_original_order`

## Integration / Functional Tests

- CLI reads sample reader JSON and emits the expected selected claim.

## Smoke Tests

- `python3 -m pytest`
- `python3 -m labclaw.claim_picker samples/reader-output.json --json`

## E2E Tests

N/A — this issue is deterministic selection after the reader.

## Manual / cURL Tests

N/A.
