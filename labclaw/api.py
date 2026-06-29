"""FastAPI surface for the LabClaw demo dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from labclaw.ledger import DEFAULT_MISSION
from labclaw.pipeline import LabPipeline

DATA_DIR = Path(os.environ.get("LABCLAW_DATA_DIR", "labclaw_data"))
FIXTURE_MODE = os.environ.get("LABCLAW_FIXTURE_MODE", "1") not in {"0", "false", "False"}

app = FastAPI(title="LabClaw API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("LABCLAW_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = LabPipeline(DATA_DIR, fixture_mode=FIXTURE_MODE)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "fixture_mode": str(FIXTURE_MODE).lower()}


@app.post("/api/demo/run")
def run_demo() -> dict:
    result = pipeline.run(mission=DEFAULT_MISSION)
    return result.to_dict()


@app.get("/api/demo/latest")
def latest_demo() -> dict:
    payload = pipeline.latest()
    if payload is None:
        return {"run": None}
    return {"run": payload}


@app.get("/api/demo/runs")
def list_runs() -> dict:
    runs = []
    for path in sorted(pipeline.runs_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        runs.append({"run_id": path.stem, "path": str(path)})
    return {"runs": runs[:20]}


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("labclaw.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
