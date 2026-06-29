# LabClaw deploy notes

## Hackathon demo (no API keys required)

The integrated demo API runs in **fixture mode** by default
(`LABCLAW_FIXTURE_MODE=1`). It uses recorded scout fixtures plus
`samples/tiny-ml-claim.md` and never calls external providers.

Required deploy env:

| Service | Variable | Value |
| --- | --- | --- |
| Railway API | `LABCLAW_FIXTURE_MODE` | `1` |
| Railway API | `LABCLAW_CORS_ORIGINS` | `*` or your Vercel URL |
| Vercel web | `NEXT_PUBLIC_API_URL` | `https://<railway-service>.up.railway.app` |

## Live mode API keys (optional, not needed for demo)

| Key | Used by | When |
| --- | --- | --- |
| `CEREBRAS_API_KEY` | Gemma reader swarm | Live multimodal claim extraction |
| `GEMINI_API_KEY` | Gemini PI | Live search plans / experiment proposals |
| `E2B_API_KEY` | E2B runner | Live sandbox experiments |
| `TELEGRAM_BOT_TOKEN` | Telegram bot | Sending pings |
| `TELEGRAM_CHAT_ID` | Telegram bot | Default chat target |

To switch the API out of demo fixtures on Railway:

```bash
LABCLAW_FIXTURE_MODE=0
```

Then add the provider keys above.

## Deploy commands

```bash
# API (repo root)
railway up --service labclaw-api

# Web (web/)
vercel --prod
```
