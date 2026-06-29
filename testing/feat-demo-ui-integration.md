# LabClaw demo integration

## Backend (Railway)

Deploy the repo root using the included `Dockerfile` / `railway.toml`.

```bash
railway up
```

Health check: `GET /health`

Run demo heartbeat: `POST /api/demo/run`

## Frontend (Vercel)

Deploy the `web/` directory.

```bash
cd web
vercel --prod
```

Set environment variable:

```text
NEXT_PUBLIC_API_URL=https://<your-railway-service>.up.railway.app
```

The UI uses the ProofSWE-inspired dark editorial theme and visualizes the full
LabClaw loop: scout → cluster → read → experiment → eval → report.
