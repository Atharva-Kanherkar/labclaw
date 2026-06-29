"use client";

import { useEffect, useState } from "react";
import { fetchLatest, runDemo, type PipelineRun } from "@/lib/api";

export default function DemoDashboard() {
  const [run, setRun] = useState<PipelineRun | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchLatest()
      .then(setRun)
      .catch((err: Error) => setError(err.message));
  }, []);

  async function handleRun() {
    setLoading(true);
    setError(null);
    try {
      const result = await runDemo();
      setRun(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run failed");
    } finally {
      setLoading(false);
    }
  }

  const markdown =
    run?.stages.find((stage) => stage.stage === "report")?.payload?.markdown ?? "";

  return (
    <div className="demo-page">
      <header className="demo-header rise" style={{ animationDelay: "0.05s" }}>
        <p className="demo-kicker">24/7 ML lab · measured evidence only</p>
        <h1 className="hero-title">LabClaw</h1>
        <p className="hero-tagline mt-4">
          source → cluster → claim → experiment → verdict
        </p>
        <p className="hero-sub mt-4">
          One heartbeat scouts research, clusters it, reads claims, runs a bounded
          experiment, and only reports when the evidence critic says it is reportable.
        </p>
        <div className="demo-actions">
          <button
            type="button"
            className="demo-button demo-button-primary"
            onClick={handleRun}
            disabled={loading}
          >
            {loading ? "Running heartbeat…" : "Run demo heartbeat"}
          </button>
          {run?.capabilities?.live_reader ? (
            <span className="chip chip-good">live Cerebras reader</span>
          ) : (
            <span className="chip chip-muted">fixture reader</span>
          )}
        </div>
        {error ? <p className="demo-error">{error}</p> : null}
      </header>

      {run ? (
        <>
          <section className="pipeline-grid rise" style={{ animationDelay: "0.12s" }}>
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

          <section className="metric-grid rise" style={{ animationDelay: "0.18s" }}>
            <div className="metric-card">
              <span>Baseline</span>
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
            style={{ animationDelay: "0.24s" }}
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
            <p className="hero-sub mt-4">{run.report.why_it_matters}</p>
            <p className="hero-sub mt-2">{run.report.recommended_next_action}</p>
          </section>

          {markdown ? (
            <pre className="report-box rise" style={{ animationDelay: "0.3s" }}>
              {markdown}
            </pre>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
