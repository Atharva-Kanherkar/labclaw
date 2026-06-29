const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export function apiBase(): string {
  return API_BASE;
}

export type DemoCapabilities = {
  live_scouts?: boolean;
  fixture_scouts?: boolean;
  live_reader?: boolean;
  live_e2b?: boolean;
  gemini_pi?: boolean;
};

export type StageProof = {
  mode?: string;
  label?: string;
  model?: string;
  provider?: string;
  source_id?: string;
  duration_ms?: number;
  api_key_suffix?: string;
  baseline_command?: string;
  candidate_command?: string;
  harness?: string;
  error?: string;
  fallback_error?: string;
};

export type PipelineRun = {
  run_id: string;
  mission: string;
  reportable: boolean;
  source: { title: string; url: string; kind: string; source_id: string };
  cluster: { cluster_id: string; topic_name: string };
  claim: {
    main_claim: string;
    benchmark_numbers?: string[];
    code_hooks?: string[];
    figures?: unknown[];
  };
  experiment_spec: {
    baseline_command: string;
    candidate_command: string;
    harness: string;
    metric: string;
  };
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
    payload?: {
      markdown?: string;
      proof?: StageProof;
      sources?: Array<{ title: string; kind: string; source_id: string; url: string }>;
    };
  }>;
  capabilities?: DemoCapabilities;
  demo_proof?: {
    started_at?: string;
    finished_at?: string;
    live_scouts_used?: boolean;
    live_cerebras_used?: boolean;
    live_e2b_used?: boolean;
    transparency?: {
      scout?: string;
      reader?: string;
      experiment?: string;
      eval?: string;
    };
    reader_proof?: StageProof;
    experiment_proof?: StageProof;
    scout_proof?: StageProof;
  };
};

export async function fetchHealth(): Promise<{
  status: string;
  fixture_mode: string;
  capabilities: DemoCapabilities;
}> {
  const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Health check failed (${response.status})`);
  return response.json();
}

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
