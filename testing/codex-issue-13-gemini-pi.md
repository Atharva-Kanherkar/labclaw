# codex/issue-13-gemini-pi — Test Contract

## Functional Behavior

- A `GeminiPI` orchestrator accepts mission text, cluster memory, new source summaries, reader claim cards, and experiment/eval results.
- The orchestrator asks an injectable client for one structured PI decision and parses it into typed Python objects.
- Outputs include a search plan, cluster priority list, experiment proposal, notify/do-not-notify decision, and final interpretation text.
- The prompt version and JSON schema version are explicit constants and included in every prompt/client call.
- The default test path uses a fake client and never requires live Gemini credentials or Google SDK imports.
- Malformed or incomplete client JSON fails with a clear `ValueError`.
- `experiment_proposal.should_run: false` does not require baseline/candidate command fields, avoiding fabricated commands for rejected weak claims.
- The optional live Gemini adapter sends the system prompt as `system_instruction` and enforces structured output with `response_schema`.

## Unit Tests

- `test_gemini_pi_builds_versioned_prompt` — prompt includes prompt/schema versions and all input sections.
- `test_gemini_pi_parses_mocked_structured_decision` — fake client response becomes typed PI decision objects.
- `test_gemini_pi_allows_declined_experiment_without_fabricated_commands` — rejected claims can omit command/metric fields.
- `test_gemini_pi_requires_experiment_commands_when_should_run` — runnable proposals still require baseline/candidate/metric/threshold.
- `test_gemini_pi_rejects_missing_required_sections` — incomplete decision JSON raises `ValueError`.
- `test_gemini_client_requires_api_key_for_live_use` — live client construction fails clearly without credentials.
- `test_gemini_api_client_uses_response_schema_and_system_instruction` — live adapter passes the documented structured-output fields to the SDK.

## Integration / Functional Tests

- `GeminiPI.decide(...)` works with a mocked client and returns search queries, cluster priorities, an experiment proposal, notify decision, and interpretation.

## Smoke Tests

- `python -m pytest tests/test_gemini_pi.py` runs without network or live Gemini credentials.

## E2E Tests

N/A — issue #13 builds the PI decision layer only. Full source-to-report E2E belongs to the parent epic once scouts, clustering, reader swarm, experiments, eval, critic, and reporting are connected.

## Manual / cURL Tests

N/A — no public CLI is required for this issue.
