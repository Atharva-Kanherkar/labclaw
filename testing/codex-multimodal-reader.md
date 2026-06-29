# codex/multimodal-reader — Test Contract

## Functional Behavior

- Empty Gemma `cards` responses must not crash human-readable CLI output.
- Null or empty Gemma message content must raise a clear `ValueError`.
- Malformed Gemma JSON must raise a clear `ValueError`.
- Oversized images must be skipped without dropping later valid images.
- Image cap must count attached images explicitly, not total content parts.
- Human-readable output must print all extracted cards, not only the first.

## Unit Tests

- `test_human_output_handles_empty_cards`
- `test_load_json_object_rejects_none_content`
- `test_load_json_object_wraps_malformed_json`
- `test_build_user_content_skips_oversized_image_and_keeps_later_valid_image`
- `test_build_user_content_counts_images_explicitly`
- `test_format_human_result_prints_all_cards`

## Integration / Functional Tests

- Existing mocked Cerebras extraction test must still pass and use `gemma-4-31b`.

## Smoke Tests

- `python3 -m pytest`
- `python3 -m labclaw.multimodal_reader samples/tiny-ml-claim.md --json --local-fixture`
- `python3 -m labclaw.multimodal_reader samples/tiny-ml-claim.md --local-fixture`

## E2E Tests

N/A — live Cerebras call requires `CEREBRAS_API_KEY` and should not run in PR tests.

## Manual / cURL Tests

N/A.
