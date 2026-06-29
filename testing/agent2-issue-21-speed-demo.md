# agent2/issue-21-speed-demo - Test Contract

## Functional Behavior

- Demo command runs against a fixture batch without live Cerebras credentials.
- Fixture mode labels the fast lane as `fixture-parser` and suppresses tok/s
  instead of presenting parser speed as a Cerebras measurement.
- Fast lane uses the reader swarm API and reports sources completed, elapsed
  time, claim cards, and per-source errors.
- Baseline lane simulates a slower provider when no live baseline is available.
- Progress events are emitted for lane start, source completion, and lane finish.
- Output includes a timing table plus the selected demo cluster.
- Demo code is isolated from the heartbeat/lab loop.

## Unit Tests

- `test_load_fixture_batch_repeats_sources_with_stable_ids`
- `test_run_speed_demo_reports_lanes_progress_and_cluster`
- `test_format_timing_table_includes_demo_metrics`
- `test_progress_json_lines_are_machine_readable`

## Integration / Functional Tests

- Speed demo uses `labclaw.multimodal_reader.read_sources` for the fast lane.
- Existing reader swarm tests from #16 continue to pass.

## Smoke Tests

- `python3 -m pytest`
- `python3 -m labclaw.speed_demo samples/tiny-ml-claim.md --repeat 2 --baseline-delay-ms 1`
- `python3 -m labclaw.speed_demo samples/tiny-ml-claim.md --repeat 2 --baseline-delay-ms 1 --json`

## E2E Tests

N/A - live Cerebras mode is available with `--live-cerebras` and
`CEREBRAS_API_KEY`, but PR tests stay on deterministic fixture mode.
