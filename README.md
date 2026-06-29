# LabClaw

LabClaw is an always-on AI scientist that fact-checks new ML/code claims by
reading the paper, looking at figures, running a small VM experiment, and
reporting whether the claim reproduces.

## Multimodal Reader MVP

Issue #2 starts with a Gemma/Cerebras reader that turns a paper/repo note into
structured claim cards:

- main claim
- figures/charts/diagrams
- benchmark numbers
- runnable code hooks

Run the sample:

```bash
export CEREBRAS_API_KEY=...
python -m labclaw.multimodal_reader samples/tiny-ml-claim.md --json
```

Run the offline fixture parser:

```bash
python -m labclaw.multimodal_reader samples/tiny-ml-claim.md --json --local-fixture
```

The live reader follows Cerebras image-input constraints:

- model: `gemma-4-31b`
- local figures are sent as base64 `image_url` content
- only PNG/JPEG figures are attached
- max 5 images per request
- max 10 MB total image payload
- response uses strict JSON schema output

Run tests:

```bash
python -m pytest
```
