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

## Telegram bot

LabClaw can talk to you on Telegram — both directions. It sends pings
(notifications) and answers commands you send the bot. It uses the raw Telegram
Bot API over HTTP with no extra dependencies.

Setup:

1. Message [@BotFather](https://t.me/BotFather) on Telegram, run `/newbot`, and
   copy the bot token it gives you.
2. Send your new bot any message so it has a chat to reply to.
3. Export the credentials (PowerShell: `$env:TELEGRAM_BOT_TOKEN = "..."`):

```bash
export TELEGRAM_BOT_TOKEN=123456:abc...      # from BotFather
export TELEGRAM_CHAT_ID=987654321            # optional default ping target
```

Start the two-way bot (long-polls for commands):

```bash
python -m labclaw.telegram        # or: labclaw-bot
```

Built-in commands: `/start`, `/help`, `/ping`, `/whoami`, and
`/read [--local] <path>`. The reader uses live Gemma extraction when
`CEREBRAS_API_KEY` is set and the Cerebras SDK is installed; pass `--local` to
force the offline parser, and if the live reader is unavailable it falls back to
offline automatically. Add your own commands with `CommandRouter.register`.

Send a one-off ping without starting the bot:

```bash
python -m labclaw.telegram --send "experiment finished"
```

From Python, e.g. to ping when a claim is verified:

```python
from labclaw.telegram import notify
notify("Claim reproduced: 2.3x speedup confirmed")
```

Notes and limitations:

- The bot long-polls and survives transient network failures (dropped
  connections, DNS hiccups, socket timeouts) with retry/backoff.
- Update handling is synchronous and single-threaded: a live `/read`
  extraction blocks other updates until it finishes (fine for one user).
- Delivery is at-most-once: if a reply send fails it is not redelivered.

Run tests:

```bash
python -m pytest
```

## Heartbeat daemon

Issue #12 adds the local run spine for the 24/7 lab loop. A single heartbeat
creates a run id and records stage transitions to an append-only JSONL ledger:

```bash
python -m labclaw.daemon --once --ledger /tmp/labclaw-ledger.jsonl
```

Resume a failed or pending run:

```bash
python -m labclaw.daemon --once --resume RUN_ID --ledger /tmp/labclaw-ledger.jsonl
```
