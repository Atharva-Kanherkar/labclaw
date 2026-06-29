const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type PipelineRun = {
  run_id: string;
  mission: string;
  reportable: boolean;
  source: { title: string; url: string; kind: string; source_id: string };
  cluster: { cluster_id: string; topic_name: string };
  claim: { main_claim: string; benchmark_numbers?: string[]; code_hooks?: string[] };
  metric_result: {
    metric: string;
    baseline: number | null;
    candidate: number | null;
    delta: number | null;
    status: string;
  };
  critic_verdict: {
    verdict: string;
    confidence: number;
    reportable: boolean;
    blocking_objections: string[];
  };
  report: {
    why_it_matters: string;
    recommended_next_action: string;
  };
  stages: Array<{
    stage: string;
    status: string;
    summary: string;
    payload?: { markdown?: string };
  }>;
};

export async function runDemo(): Promise<PipelineRun> {
  const response = await fetch(`${API_BASE}/api/demo/run`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Run failed (${response.status})`);
  }
  return response.json();
}

export async function fetchLatest(): Promise<PipelineRun | null> {
  const response = await fetch(`${API_BASE}/api/demo/latest`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Latest fetch failed (${response.status})`);
  }
  const payload = await response.json();
  return payload.run ?? null;
}
