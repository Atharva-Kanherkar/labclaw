import json

import pytest

from labclaw.gemini_pi import (
    DEFAULT_PI_MODEL,
    PI_PROMPT_VERSION,
    PI_SCHEMA_VERSION,
    GeminiAPIClient,
    GeminiPI,
    build_pi_messages,
    PIInputs,
)


class FakePIClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, *, messages, schema, model):
        self.calls.append({"messages": messages, "schema": schema, "model": model})
        return self.response


def sample_decision():
    return {
        "schema_version": PI_SCHEMA_VERSION,
        "search_plan": {
            "queries": ["small language model cache-aware attention", "tiny transformer inference speed"],
            "source_kinds": ["paper", "repo"],
            "rationale": "Find bounded speed claims with code hooks.",
        },
        "cluster_priorities": [
            {
                "cluster_id": "cluster-speed",
                "topic": "inference speed / kernels",
                "priority": 92,
                "rationale": "Fresh claims include runnable benchmarks.",
            }
        ],
        "experiment_proposal": {
            "claim_id": "claim-cache-aware",
            "cluster_id": "cluster-speed",
            "should_run": True,
            "goal": "Compare baseline attention against cache-aware attention.",
            "baseline_command": "python bench.py --mode baseline",
            "candidate_command": "python bench.py --mode cache-aware",
            "metric": "tokens_per_second",
            "threshold": "candidate >= baseline * 1.10",
            "rationale": "The claim has explicit commands and a measurable throughput metric.",
        },
        "notification_decision": {
            "notify": False,
            "confidence": 0.62,
            "reason": "Wait for measured E2B results before interrupting the user.",
        },
        "interpretation": "Promising but not reportable until baseline/candidate metrics exist.",
    }


def test_gemini_pi_builds_versioned_prompt() -> None:
    messages = build_pi_messages(
        PIInputs(
            mission="Track cheap ML speedups.",
            cluster_memory=[{"cluster_id": "cluster-speed", "topic": "inference speed"}],
            source_summaries=[{"source_id": "src-1", "title": "Fast Attention"}],
            claim_cards=[{"id": "claim-cache-aware", "main_claim": "1.4x faster attention"}],
            experiment_results=[{"claim_id": "claim-old", "verdict": "refuted"}],
        )
    )

    payload = json.loads(messages[1]["content"])
    assert messages[0]["role"] == "system"
    assert payload["prompt_version"] == PI_PROMPT_VERSION
    assert payload["schema_version"] == PI_SCHEMA_VERSION
    assert payload["mission"] == "Track cheap ML speedups."
    assert payload["cluster_memory"][0]["cluster_id"] == "cluster-speed"
    assert payload["source_summaries"][0]["title"] == "Fast Attention"
    assert payload["claim_cards"][0]["id"] == "claim-cache-aware"
    assert payload["experiment_results"][0]["verdict"] == "refuted"
    assert "experiment_proposal" in payload["required_outputs"]


def test_gemini_pi_parses_mocked_structured_decision() -> None:
    fake_client = FakePIClient(sample_decision())
    pi = GeminiPI(fake_client)

    decision = pi.decide(
        mission="Find measurable inference improvements.",
        cluster_memory=[{"cluster_id": "cluster-speed", "topic": "inference speed"}],
        source_summaries=[{"source_id": "src-1", "title": "Fast Attention"}],
        claim_cards=[{"id": "claim-cache-aware", "main_claim": "1.4x faster attention"}],
        experiment_results=[],
    )

    call = fake_client.calls[0]
    assert call["model"] == DEFAULT_PI_MODEL
    assert call["schema"]["required"] == [
        "schema_version",
        "search_plan",
        "cluster_priorities",
        "experiment_proposal",
        "notification_decision",
        "interpretation",
    ]
    assert decision.search_plan.queries[0] == "small language model cache-aware attention"
    assert decision.cluster_priorities[0].priority == 92
    assert decision.experiment_proposal.baseline_command == "python bench.py --mode baseline"
    assert decision.experiment_proposal.candidate_command == "python bench.py --mode cache-aware"
    assert decision.notification_decision.notify is False
    assert decision.interpretation.startswith("Promising")


def test_gemini_pi_accepts_json_string_from_client() -> None:
    pi = GeminiPI(FakePIClient(json.dumps(sample_decision())))

    decision = pi.decide(mission="Track code claims.")

    assert decision.schema_version == PI_SCHEMA_VERSION
    assert decision.experiment_proposal.should_run is True


def test_gemini_pi_rejects_missing_required_sections() -> None:
    pi = GeminiPI(FakePIClient({"schema_version": PI_SCHEMA_VERSION}))

    with pytest.raises(ValueError, match="missing required field"):
        pi.decide(mission="Track code claims.")


def test_gemini_pi_rejects_malformed_json() -> None:
    pi = GeminiPI(FakePIClient("{not json"))

    with pytest.raises(ValueError, match="malformed JSON"):
        pi.decide(mission="Track code claims.")


def test_gemini_client_requires_api_key_for_live_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiAPIClient()
