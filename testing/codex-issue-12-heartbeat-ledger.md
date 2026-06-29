# codex/issue-12-heartbeat-ledger — Test Contract

## Functional Behavior

- `labclaw daemon --once --ledger PATH` creates one heartbeat run with a stable `run_id`.
- A heartbeat records stage transitions for `scout`, `cluster`, `read`, `experiment`, `eval`, and `report`.
- Successful runs end with heartbeat status `succeeded` and every default stage status `succeeded`.
- Failed stages are represented in the ledger with status `failed`, an error message, and heartbeat status `failed`.
- `labclaw daemon --once --resume RUN_ID --ledger PATH` retries the first failed or pending stage for that run instead of creating a new heartbeat.
- The ledger is local JSONL storage, with a storage-agnostic Python interface for future SQLite replacement.

## Unit Tests

- `test_run_ledger_creates_heartbeat` — new heartbeat has mission, status, timestamps, and stage records.
- `test_run_ledger_records_stage_failure` — failed stage stores error details and marks the heartbeat failed.
- `test_heartbeat_daemon_successful_once_run` — default stages all succeed with local no-op handlers.
- `test_heartbeat_daemon_resumes_failed_run` — resume targets an existing run and completes the failed stage.

## Integration / Functional Tests

- CLI invocation with `--once --ledger PATH` exits zero and writes JSONL records.
- CLI invocation with `--once --resume RUN_ID --ledger PATH` exits zero when a retryable run exists.

## Smoke Tests

- `python -m labclaw.daemon --once --ledger /tmp/labclaw-ledger.jsonl` creates a run without external credentials.

## E2E Tests

N/A — issue #12 only creates the local heartbeat spine. Full source-to-report E2E belongs to the parent epic once scouts, readers, experiments, evals, and reports are connected.

## Manual / cURL Tests

```bash
python -m labclaw.daemon --once --ledger /tmp/labclaw-ledger.jsonl
python -m labclaw.daemon --once --resume RUN_ID --ledger /tmp/labclaw-ledger.jsonl
```
