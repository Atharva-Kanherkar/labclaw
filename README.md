# 🦞 LabClaw

> An always-on AI scientist that fact-checks new ML/code claims — it reads the paper, looks at the figures, runs a small sandboxed experiment, and tells you whether the claim actually reproduces.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

---

## What is LabClaw?

Every day, new papers and repos claim "2.3× faster," "+5 points on the benchmark," "drop-in replacement, no quality loss." Most of those claims are never independently checked.

**LabClaw is a 24/7 autonomous research loop that checks them for you.** It continuously scouts fresh ML/code claims, extracts what's actually being asserted (including from figures and charts), proposes a minimal experiment to test it, runs that experiment in a bounded sandbox, and then applies a strict evidence gate before it tells you anything. If a claim reproduces, you get a ping. If it doesn't, you get told that too — with the receipts.

It runs out of the box in **fixture mode** (no API keys, fully deterministic) so you can try the whole pipeline locally, then flip on live providers when you're ready.

## How it works

LabClaw is a pipeline of small, independently-testable stages wired into a heartbeat loop:

```
  Scouts ──▶ Clustering ──▶ Multimodal Reader ──▶ Gemini PI ──▶ Eval Harness ──▶ E2B Runner ──▶ Evidence Critic ──▶ Report / Telegram
 (arXiv,                    (Gemma via         (strategic     (defines what    (bounded         (truth gate:
  GitHub)                    Cerebras —         orchestrator:  "improved"       sandbox          reproduced?
                             claim cards from   plans the      means)           experiment)      refuted?
                             text + figures)    experiment)                                       inconclusive?)
```

| Stage | Module | What it does |
| --- | --- | --- |
| **Scouts** | `sources.py` | Pull fresh candidate claims from arXiv and GitHub, dedupe against what's been seen. |
| **Clustering** | `clustering.py` | Group related sources into topic clusters so the PI can prioritize. |
| **Multimodal Reader** | `multimodal_reader.py` | A Gemma reader swarm (via Cerebras) turns a paper/repo into structured **claim cards**: the main claim, figures/charts, benchmark numbers, and runnable code hooks. |
| **Gemini PI** | `gemini_pi.py` | The "Principal Investigator." Turns mission + context into a decision: search plan, cluster priorities, an experiment proposal, and whether to notify. Rejects weak claims instead of inventing experiments for them. |
| **Eval Harness** | `eval_harness.py` | The local truth contract for "improved" — metric, direction, threshold, baseline vs. candidate commands. Supports absolute deltas (`delta>=5`) and ratios (`candidate >= baseline * 1.10`). |
| **E2B Runner** | `e2b_runner.py` | Runs setup/baseline/candidate commands in a bounded E2B sandbox, captures logs, parses metric JSON, and downloads declared artifacts. Rejects unbounded shell control tokens. |
| **Evidence Critic** | `evidence_critic.py` | The gate before anything is reported. Only `reproduced` results with no blocking objections and enough confidence become `reportable`. Worse candidates are `refuted`; below-threshold nulls are `inconclusive`; flaky timeouts trigger `rerun_needed`. |
| **Heartbeat daemon** | `daemon.py` | The 24/7 spine. Each heartbeat creates a run id and records stage transitions to an append-only JSONL ledger, so runs can be resumed. |
| **Telegram bot** | `telegram.py` | Two-way notifications and commands over the raw Telegram Bot API (no extra deps). |
| **API + Web** | `api.py`, `web/` | A FastAPI surface and a Next.js dashboard for running the demo pipeline and inspecting runs. |

Every stage is designed to run with an injected fake/fixture, so the whole thing is testable without a single API key.

## Quickstart

```bash
git clone https://github.com/Atharva-Kanherkar/labclaw.git
cd labclaw
pip install -e ".[dev]"
```

Run the full pipeline offline (fixture mode — deterministic, no keys needed):

```bash
python -m labclaw.speed_demo samples/tiny-ml-claim.md --repeat 4
```

Read a claim into structured claim cards using the offline parser:

```bash
python -m labclaw.multimodal_reader samples/tiny-ml-claim.md --json --local-fixture
```

Run a single heartbeat of the 24/7 loop:

```bash
labclaw daemon --once --ledger /tmp/labclaw-ledger.jsonl
# resume a previous run:
labclaw daemon --once --resume RUN_ID --ledger /tmp/labclaw-ledger.jsonl
```

Scout for fresh sources:

```bash
labclaw scout --once --max 25 --data-dir labclaw_data
```

Run the tests:

```bash
python -m pytest
```

## Going live

LabClaw works fully in fixture mode. To use live providers, install the optional
extras and set the relevant keys — each capability is independent, so you can
enable just the ones you want.

```bash
pip install -e ".[gemini,e2b,figures]"
```

| Variable | Powers | Used by |
| --- | --- | --- |
| `CEREBRAS_API_KEY` | Live Gemma multimodal reader swarm | `multimodal_reader.py` |
| `GEMINI_API_KEY` | Live PI: search plans & experiment proposals | `gemini_pi.py` |
| `E2B_API_KEY` | Live sandboxed experiments | `e2b_runner.py` |
| `TELEGRAM_BOT_TOKEN` | Sending/answering Telegram messages | `telegram.py` |
| `TELEGRAM_CHAT_ID` | Default ping target (optional) | `telegram.py` |

The live multimodal reader follows Cerebras image-input constraints: model
`gemma-4-31b`, local figures sent as base64 `image_url` content, PNG/JPEG only,
max 5 images and 10 MB per request, strict JSON-schema output. For batch reads,
prefer `read_sources(..., client_factory=...)` so each worker gets its own client.

### Telegram bot

LabClaw can talk to you both directions on Telegram — pings out, commands in.

1. Message [@BotFather](https://t.me/BotFather), run `/newbot`, copy the token.
2. Send your new bot any message so it has a chat to reply to.
3. Export credentials and start the bot:

```bash
export TELEGRAM_BOT_TOKEN=123456:abc...
export TELEGRAM_CHAT_ID=987654321   # optional
python -m labclaw.telegram          # or: labclaw-bot
```

Built-in commands: `/start`, `/help`, `/ping`, `/whoami`, `/read [--local] <path>`.
Register your own with `CommandRouter.register`. Send a one-off ping without
starting the bot:

```bash
python -m labclaw.telegram --send "experiment finished"
```

Or from Python:

```python
from labclaw.telegram import notify
notify("Claim reproduced: 2.3x speedup confirmed")
```

## Usage from Python

The PI, eval harness, and evidence critic all accept injected fakes, so you can
drive any stage directly:

```python
from labclaw.eval_harness import default_registry, spec_from_pi_proposal
from labclaw.evidence_critic import EvidenceCritic, EvidenceInput, ReproducibilityContext

spec = spec_from_pi_proposal({
    "claim_id": "claim-cache-aware",
    "cluster_id": "cluster-speed",
    "should_run": True,
    "goal": "Compare throughput.",
    "baseline_command": "metric:tokens_per_second=42",
    "candidate_command": "metric:tokens_per_second=55",
    "metric": "tokens_per_second",
    "threshold": "delta>=5",
    "rationale": "Bounded speed claim with explicit commands.",
})

metric = default_registry().run(spec)
verdict = EvidenceCritic().evaluate(
    EvidenceInput(
        spec=spec,
        metric_result=metric,
        run_status="succeeded",
        reproducibility=ReproducibilityContext(random_seed=42, sample_size=128),
    )
)
print(verdict)  # reproduced / refuted / inconclusive / rerun_needed
```

## Deployment

The integrated demo deploys with the API in fixture mode (no keys required) and
a Next.js dashboard. The bundled `Dockerfile` runs the API; see
[`DEPLOY.md`](DEPLOY.md) for Railway + Vercel env config and commands.

```bash
docker build -t labclaw . && docker run -p 8000:8000 labclaw
# then open http://localhost:8000/health
```

## Project layout

```
labclaw/        # core package — one module per pipeline stage
web/            # Next.js demo dashboard
samples/        # sample claim used by the offline demo
tests/          # pytest suite (fixtures, fakes — no keys needed)
DEPLOY.md       # Railway + Vercel deploy notes
Dockerfile      # API container (fixture mode by default)
```

## Contributing

Contributions are welcome! LabClaw is built as a set of small, independently
testable stages — the easiest way in is to pick one module and improve it.

1. Fork the repo and create a feature branch.
2. Add or update tests in `tests/` — every stage is testable with fakes/fixtures, no API keys required.
3. Run `python -m pytest` and make sure it's green.
4. Open a PR describing the change.

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.
