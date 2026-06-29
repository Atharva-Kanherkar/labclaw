# agent2/issue-16-reader-swarm - Test Contract

## Functional Behavior

- Reader accepts in-memory source text without requiring a file path.
- Reader consumes `SourceRecord`-shaped fixtures with stable `source_id`,
  `raw_text`, local `figures`, and metadata.
- Batch reader returns one ordered result object per input source.
- Per-source extraction failures are captured on that source and do not fail the
  whole batch.
- `max_workers` limits concurrent source processing.
- Each source result includes elapsed timing, card count, and error metadata.

## Unit Tests

- `test_read_source_text_accepts_in_memory_source_title`
- `test_source_record_figures_are_attached_for_gemma`
- `test_read_sources_preserves_order_and_isolates_failures`
- `test_read_sources_respects_configured_concurrency`

## Integration / Functional Tests

- Existing single-source mocked Cerebras extraction still uses `gemma-4-31b`.
- Existing local fixture and image constraint tests still pass.

## Smoke Tests

- `python3 -m pytest`
- `python3 -m labclaw.multimodal_reader samples/tiny-ml-claim.md --json --local-fixture`

## E2E Tests

N/A - live Cerebras calls require `CEREBRAS_API_KEY` and should not run in PR
tests.
