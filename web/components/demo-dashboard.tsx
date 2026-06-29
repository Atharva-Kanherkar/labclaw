"use client";

import { useEffect, useState } from "react";
import {
  fetchHealth,
  fetchLatest,
  runDemo,
  type DemoCapabilities,
  type PipelineRun,
} from "@/lib/api";

const LIVE_STEPS = [
  "Loading recorded arXiv + GitHub scouts…",
  "Clustering source into topic memory…",
  "Calling Cerebras Gemma reader on sample paper…",
  "Running baseline vs candidate experiment…",
  "Scoring metrics + evidence critic…",
  "Building report…",
];

export default function DemoDashboard() {
  const [run, setRun] = useState<PipelineRun | null>(null);
  const [caps, setCaps] = useState<DemoCapabilities | null>(null);
  const [loading, setLoading] = useState(false);
  const [liveStep, setLiveStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((health) => setCaps(health.capabilities))
      .catch(() => setCaps(null));
    fetchLatest()
      .then(setRun)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!loading) return;
    const timer = setInterval(() => {
      setLiveStep((step) => (step + 1) % LIVE_STEPS.length);
    }, 1400);
    return () => clearInterval(timer);
  }, [loading]);

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
  const scoutSources = run?.stages.find((stage) => stage.stage === "scout")?.payload?.sources ?? [];
  const readProof = readStage?.payload?.proof;
  const experimentProof = run?.stages.find((stage) => stage.stage === "experiment")?.payload?.proof;
  const markdown =
    run?.stages.find((stage) => stage.stage === "report")?.payload?.markdown ?? "";
  const liveUsed = run?.demo_proof?.live_cerebras_used ?? readProof?.mode === "live_cerebras";

  return (
    <div className="demo-page">
      <header className="demo-header rise" style={{ animationDelay: "0.05s" }}>
        <p className="demo-kicker">24/7 ML lab · measured evidence only</p>
        <h1 className="hero-title">LabClaw</h1>
        <p className="hero-tagline mt-4">
          source → cluster → claim → experiment → verdict
        </p>

        <div className={`proof-banner mt-6 ${liveUsed ? "proof-banner-live" : ""}`}>
          <strong>{liveUsed ? "LIVE CEREBRAS READER ACTIVE" : "FIXTURE DEMO MODE"}</strong>
          <span>
            {caps?.live_reader
              ? "CEREBRAS_API_KEY detected · gemma-4-31b will run on heartbeat"
              : "Set CEREBRAS_API_KEY on Railway to enable live reader"}
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
          {caps?.live_reader ? (
            <span className="chip chip-good">Cerebras key loaded</span>
          ) : (
            <span className="chip chip-bad">no Cerebras key</span>
          )}
          <span className="chip chip-muted">scouts: fixtures</span>
          <span className="chip chip-muted">eval: harness</span>
        </div>

        {loading ? (
          <div className="activity-log mt-4">
            <p className="activity-log-title">Live activity</p>
            <p className="activity-log-line">{LIVE_STEPS[liveStep]}</p>
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
              {run.demo_proof?.live_cerebras_used ? (
                <span className="chip chip-good">live cerebras verified</span>
              ) : (
                <span className="chip chip-bad">fixture reader used</span>
              )}
              <span className="chip">started {run.demo_proof?.started_at ?? "—"}</span>
            </div>
            <ul className="transparency-list">
              <li>{run.demo_proof?.transparency?.scout}</li>
              <li>{run.demo_proof?.transparency?.reader}</li>
              <li>{run.demo_proof?.transparency?.experiment}</li>
            </ul>
          </section>

          <section className="detail-card rise" style={{ animationDelay: "0.12s" }}>
            <p className="demo-kicker">Scout inbox</p>
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
