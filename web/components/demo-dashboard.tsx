"use client";

import { useEffect, useState } from "react";
import {
  fetchHealth,
  fetchLatest,
  runDemo,
  type DemoCapabilities,
  type PipelineRun,
} from "@/lib/api";

function liveSteps(caps: DemoCapabilities | null) {
  return [
    caps?.live_scouts ? "Fetching live arXiv + GitHub scouts…" : "Loading offline scout fixtures…",
    "Clustering source into topic memory…",
    caps?.live_reader ? "Calling Cerebras on scouted source text…" : "Parsing scouted source locally…",
    caps?.live_e2b ? "Running measured baseline vs candidate in E2B…" : "Running local metric harness…",
    "Scoring metrics + evidence critic…",
    "Building report…",
  ];
}

export default function DemoDashboard() {
  const [run, setRun] = useState<PipelineRun | null>(null);
  const [caps, setCaps] = useState<DemoCapabilities | null>(null);
  const [fixtureMode, setFixtureMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [liveStep, setLiveStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((health) => {
        setCaps(health.capabilities);
        setFixtureMode(health.fixture_mode === "true");
      })
      .catch(() => setCaps(null));
    fetchLatest()
      .then(setRun)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!loading) return;
    const steps = liveSteps(caps);
    const timer = setInterval(() => {
      setLiveStep((step) => (step + 1) % steps.length);
    }, 1400);
    return () => clearInterval(timer);
  }, [loading, caps]);

  async function handleRun() {
    setLoading(true);
    setLiveStep(0);
    setError(null);
    try {
      const result = await runDemo();
      setRun(result);
      setCaps(result.capabilities ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run failed");
    } finally {
      setLoading(false);
    }
  }

  const readStage = run?.stages.find((stage) => stage.stage === "read");
  const scoutStage = run?.stages.find((stage) => stage.stage === "scout");
  const scoutSources = scoutStage?.payload?.sources ?? [];
  const readProof = readStage?.payload?.proof ?? run?.demo_proof?.reader_proof;
  const scoutProof = scoutStage?.payload?.proof ?? run?.demo_proof?.scout_proof;
  const experimentProof =
    run?.stages.find((stage) => stage.stage === "experiment")?.payload?.proof ??
    run?.demo_proof?.experiment_proof;
  const markdown =
    run?.stages.find((stage) => stage.stage === "report")?.payload?.markdown ?? "";

  const liveScouts = run?.demo_proof?.live_scouts_used ?? scoutProof?.mode === "live_network";
  const liveReader = run?.demo_proof?.live_cerebras_used ?? readProof?.mode === "live_cerebras";
  const liveE2b = run?.demo_proof?.live_e2b_used ?? experimentProof?.mode === "live_e2b";
  const fullyLive = liveScouts && liveReader && liveE2b;
  const activitySteps = liveSteps(caps);

  return (
    <div className="demo-page">
      <header className="demo-header rise" style={{ animationDelay: "0.05s" }}>
        <p className="demo-kicker">24/7 ML lab · measured evidence only</p>
        <h1 className="hero-title">LabClaw</h1>
        <p className="hero-tagline mt-4">
          source → cluster → claim → experiment → verdict
        </p>

        <div className={`proof-banner mt-6 ${fullyLive ? "proof-banner-live" : ""}`}>
          <strong>
            {fullyLive
              ? "LIVE DEMO — network scouts · Cerebras · E2B"
              : fixtureMode
                ? "PARTIAL LIVE — fixture scouts enabled on API"
                : "MIXED MODE — see proof chips below"}
          </strong>
          <span>
            {caps?.live_reader
              ? "Cerebras reader available"
              : "Set CEREBRAS_API_KEY on Railway"}
            {" · "}
            {caps?.live_e2b ? "E2B sandbox available" : "Set E2B_API_KEY + LABCLAW_LIVE_E2B=1"}
            {" · "}
            {caps?.live_scouts ? "live scouts default" : "LABCLAW_FIXTURE_MODE=1 forces offline scouts"}
          </span>
        </div>

        <div className="demo-actions mt-4">
          <button
            type="button"
            className="demo-button demo-button-primary"
            onClick={handleRun}
            disabled={loading}
          >
            {loading ? "Running heartbeat…" : "Run demo heartbeat"}
          </button>
          {caps?.live_scouts ? (
            <span className="chip chip-good">live scouts</span>
          ) : (
            <span className="chip chip-bad">fixture scouts</span>
          )}
          {caps?.live_reader ? (
            <span className="chip chip-good">Cerebras key loaded</span>
          ) : (
            <span className="chip chip-bad">no Cerebras key</span>
          )}
          {caps?.live_e2b ? (
            <span className="chip chip-good">E2B live</span>
          ) : (
            <span className="chip chip-muted">E2B off / harness</span>
          )}
        </div>

        {loading ? (
          <div className="activity-log mt-4">
            <p className="activity-log-title">Live activity</p>
            <p className="activity-log-line">{activitySteps[liveStep]}</p>
            <div className="activity-bar">
              <span className="activity-bar-fill" />
            </div>
          </div>
        ) : null}
        {error ? <p className="demo-error">{error}</p> : null}
      </header>

      {run ? (
        <>
          <section className="detail-card rise" style={{ animationDelay: "0.1s" }}>
            <p className="demo-kicker">Run proof</p>
            <h2 className="detail-title">{run.run_id}</h2>
            <div className="chip-row">
              {liveScouts ? (
                <span className="chip chip-good">live scouts verified</span>
              ) : (
                <span className="chip chip-bad">offline scout fixtures</span>
              )}
              {liveReader ? (
                <span className="chip chip-good">live cerebras verified</span>
              ) : (
                <span className="chip chip-bad">fixture reader used</span>
              )}
              {liveE2b ? (
                <span className="chip chip-good">live E2B measured</span>
              ) : (
                <span className="chip chip-muted">local harness / E2B fallback</span>
              )}
              <span className="chip">started {run.demo_proof?.started_at ?? "—"}</span>
            </div>
            <ul className="transparency-list">
              <li>{run.demo_proof?.transparency?.scout ?? scoutProof?.label}</li>
              <li>{run.demo_proof?.transparency?.reader ?? readProof?.label}</li>
              <li>{run.demo_proof?.transparency?.experiment ?? experimentProof?.label}</li>
            </ul>
          </section>

          <section className="detail-card rise" style={{ animationDelay: "0.12s" }}>
            <p className="demo-kicker">Scout inbox</p>
            <p className="hero-sub">{scoutStage?.summary}</p>
            <div className="source-table-wrap">
              <table className="source-table">
                <thead>
                  <tr>
                    <th>Source</th>
                    <th>Kind</th>
                    <th>ID</th>
                  </tr>
                </thead>
                <tbody>
                  {scoutSources.slice(0, 6).map((source: { title: string; kind: string; source_id: string; url: string }) => (
                    <tr key={source.source_id}>
                      <td>
                        <a href={source.url} target="_blank" rel="noreferrer">
                          {source.title}
                        </a>
                      </td>
                      <td>{source.kind}</td>
                      <td>{source.source_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="detail-card rise" style={{ animationDelay: "0.14s" }}>
            <p className="demo-kicker">Cerebras reader output</p>
            <h2 className="detail-title">{readStage?.summary}</h2>
            {readProof ? (
              <div className="proof-grid">
                <div><span>mode</span><strong>{readProof.mode}</strong></div>
                <div><span>model</span><strong>{readProof.model ?? "fixture"}</strong></div>
                <div><span>source</span><strong>{readProof.source_id ?? run.source.source_id}</strong></div>
                <div><span>duration</span><strong>{readProof.duration_ms ?? "—"} ms</strong></div>
                <div><span>key suffix</span><strong>{readProof.api_key_suffix ?? "n/a"}</strong></div>
              </div>
            ) : null}
            <p className="claim-box mt-4">{run.claim.main_claim}</p>
            <div className="chip-row mt-3">
              {(run.claim.benchmark_numbers ?? []).map((n) => (
                <span className="chip" key={n}>{n}</span>
              ))}
            </div>
            <pre className="code-box mt-3">
              {(run.claim.code_hooks ?? []).join("\n") || "no code hooks"}
            </pre>
          </section>

          <section className="pipeline-grid rise" style={{ animationDelay: "0.16s" }}>
            {run.stages.map((stage) => (
              <div className="stage-row" key={stage.stage}>
                <div className="stage-label">{stage.stage}</div>
                <div>
                  <div className="stage-title">{stage.summary}</div>
                  <div className="stage-summary">{stage.status}</div>
                </div>
              </div>
            ))}
          </section>

          <section className="detail-card rise" style={{ animationDelay: "0.18s" }}>
            <p className="demo-kicker">Experiment commands</p>
            <pre className="code-box">
              baseline: {experimentProof?.baseline_command ?? run.experiment_spec.baseline_command}
              {"\n"}
              candidate: {experimentProof?.candidate_command ?? run.experiment_spec.candidate_command}
            </pre>
            <p className="hero-sub mt-3">{experimentProof?.label}</p>
          </section>

          <section className="metric-grid rise" style={{ animationDelay: "0.2s" }}>
            <div className="metric-card">
              <span>Baseline {run.metric_result.metric}</span>
              <strong>{run.metric_result.baseline ?? "—"}</strong>
            </div>
            <div className="metric-card">
              <span>Candidate</span>
              <strong>{run.metric_result.candidate ?? "—"}</strong>
            </div>
            <div className="metric-card">
              <span>Delta</span>
              <strong>{run.metric_result.delta ?? "—"}</strong>
            </div>
          </section>

          <section
            className={`verdict-card rise ${run.reportable ? "verdict-good" : "verdict-bad"}`}
            style={{ animationDelay: "0.22s" }}
          >
            <p className="demo-kicker">Evidence critic</p>
            <h2 className="hero-tagline mt-2">{run.critic_verdict.verdict}</h2>
            <div className="chip-row">
              <span className={`chip ${run.reportable ? "chip-good" : "chip-bad"}`}>
                reportable: {String(run.reportable)}
              </span>
              <span className="chip">confidence {run.critic_verdict.confidence.toFixed(2)}</span>
              <span className="chip">{run.cluster.topic_name}</span>
            </div>
          </section>

          {markdown ? (
            <pre className="report-box rise" style={{ animationDelay: "0.24s" }}>
              {markdown}
            </pre>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
