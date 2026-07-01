"""OpenAI principal investigator orchestration for LabClaw."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

from labclaw.openai_client import DEFAULT_OPENAI_MODEL, OpenAIClient, load_json_object

PI_PROMPT_VERSION = "openai-pi-v1"
PI_SCHEMA_VERSION = 1
DEFAULT_PI_MODEL = DEFAULT_OPENAI_MODEL

SYSTEM_PROMPT = """You are LabClaw's OpenAI principal investigator.
You own strategy, not raw reader throughput. Convert the standing mission into
search plans, prioritize clusters, choose only bounded experiments, reject weak
claims before VM spend, and write concise user-facing interpretation after
evidence exists. Return only JSON that matches the provided schema."""

PI_DECISION_SCHEMA = {
    "title": "pi_decision",
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer"},
        "search_plan": {
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}},
                "source_kinds": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["paper", "repo", "model", "benchmark", "blog"]},
                },
                "rationale": {"type": "string"},
            },
            "required": ["queries", "source_kinds", "rationale"],
            "additionalProperties": False,
        },
        "cluster_priorities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "string"},
                    "topic": {"type": "string"},
                    "priority": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["cluster_id", "topic", "priority", "rationale"],
                "additionalProperties": False,
            },
        },
        "experiment_proposal": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "cluster_id": {"type": "string"},
                "should_run": {"type": "boolean"},
                "goal": {"type": "string"},
                "baseline_command": {"type": "string"},
                "candidate_command": {"type": "string"},
                "metric": {"type": "string"},
                "threshold": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["claim_id", "cluster_id", "should_run", "goal", "rationale"],
            "additionalProperties": False,
        },
        "notification_decision": {
            "type": "object",
            "properties": {
                "notify": {"type": "boolean"},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["notify", "confidence", "reason"],
            "additionalProperties": False,
        },
        "interpretation": {"type": "string"},
    },
    "required": [
        "schema_version",
        "search_plan",
        "cluster_priorities",
        "experiment_proposal",
        "notification_decision",
        "interpretation",
    ],
    "additionalProperties": False,
}


class PIDecisionClient(Protocol):
    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        model: str,
    ) -> Mapping[str, Any] | str:
        ...


@dataclass(frozen=True)
class SearchPlan:
    queries: list[str]
    source_kinds: list[str]
    rationale: str


@dataclass(frozen=True)
class ClusterPriority:
    cluster_id: str
    topic: str
    priority: int
    rationale: str


@dataclass(frozen=True)
class ExperimentProposal:
    claim_id: str
    cluster_id: str
    should_run: bool
    goal: str
    baseline_command: str
    candidate_command: str
    metric: str
    threshold: str
    rationale: str


@dataclass(frozen=True)
class NotificationDecision:
    notify: bool
    confidence: float
    reason: str


@dataclass(frozen=True)
class PIDecision:
    schema_version: int
    search_plan: SearchPlan
    cluster_priorities: list[ClusterPriority]
    experiment_proposal: ExperimentProposal
    notification_decision: NotificationDecision
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PIInputs:
    mission: str
    cluster_memory: Sequence[Mapping[str, Any]]
    source_summaries: Sequence[Mapping[str, Any]]
    claim_cards: Sequence[Mapping[str, Any]]
    experiment_results: Sequence[Mapping[str, Any]]


class OpenAIPI:
    """Strategic PI wrapper with injectable structured-output client."""

    def __init__(self, client: PIDecisionClient, *, model: str = DEFAULT_PI_MODEL) -> None:
        self.client = client
        self.model = model

    def decide(
        self,
        *,
        mission: str,
        cluster_memory: Sequence[Mapping[str, Any]] = (),
        source_summaries: Sequence[Mapping[str, Any]] = (),
        claim_cards: Sequence[Mapping[str, Any]] = (),
        experiment_results: Sequence[Mapping[str, Any]] = (),
    ) -> PIDecision:
        inputs = PIInputs(
            mission=mission,
            cluster_memory=cluster_memory,
            source_summaries=source_summaries,
            claim_cards=claim_cards,
            experiment_results=experiment_results,
        )
        raw = self.client.complete_json(
            messages=build_pi_messages(inputs),
            schema=PI_DECISION_SCHEMA,
            model=self.model,
        )
        return decision_from_payload(load_json_object(raw, provider="OpenAI PI"))


class OpenAIAPIClient(OpenAIClient):
    """Alias for tests and live PI use."""


def build_pi_messages(inputs: PIInputs) -> list[dict[str, str]]:
    payload = {
        "prompt_version": PI_PROMPT_VERSION,
        "schema_version": PI_SCHEMA_VERSION,
        "mission": inputs.mission,
        "cluster_memory": list(inputs.cluster_memory),
        "source_summaries": list(inputs.source_summaries),
        "claim_cards": list(inputs.claim_cards),
        "experiment_results": list(inputs.experiment_results),
        "required_outputs": [
            "search_plan",
            "cluster_priorities",
            "experiment_proposal",
            "notification_decision",
            "interpretation",
        ],
        "anti_slop_rules": [
            "Do not propose open-ended shell autonomy.",
            "Only propose experiments with explicit baseline and candidate commands.",
            "Do not notify the user unless measured evidence is strong enough.",
            "Prefer cheap, bounded experiments before E2B spend.",
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
    ]


def decision_from_payload(payload: Mapping[str, Any]) -> PIDecision:
    require_keys(payload, PI_DECISION_SCHEMA["required"], "PI decision")
    search_plan = payload["search_plan"]
    experiment = payload["experiment_proposal"]
    notification = payload["notification_decision"]
    require_keys(search_plan, PI_DECISION_SCHEMA["properties"]["search_plan"]["required"], "search_plan")
    require_keys(experiment, PI_DECISION_SCHEMA["properties"]["experiment_proposal"]["required"], "experiment_proposal")
    if bool(experiment["should_run"]):
        require_keys(
            experiment,
            ["baseline_command", "candidate_command", "metric", "threshold"],
            "experiment_proposal",
        )
    require_keys(notification, PI_DECISION_SCHEMA["properties"]["notification_decision"]["required"], "notification_decision")
    for index, cluster in enumerate(payload["cluster_priorities"]):
        require_keys(
            cluster,
            PI_DECISION_SCHEMA["properties"]["cluster_priorities"]["items"]["required"],
            f"cluster_priorities[{index}]",
        )

    return PIDecision(
        schema_version=int(payload["schema_version"]),
        search_plan=SearchPlan(
            queries=[str(query) for query in search_plan["queries"]],
            source_kinds=[str(kind) for kind in search_plan["source_kinds"]],
            rationale=str(search_plan["rationale"]),
        ),
        cluster_priorities=[
            ClusterPriority(
                cluster_id=str(cluster["cluster_id"]),
                topic=str(cluster["topic"]),
                priority=int(cluster["priority"]),
                rationale=str(cluster["rationale"]),
            )
            for cluster in payload["cluster_priorities"]
        ],
        experiment_proposal=ExperimentProposal(
            claim_id=str(experiment["claim_id"]),
            cluster_id=str(experiment["cluster_id"]),
            should_run=bool(experiment["should_run"]),
            goal=str(experiment["goal"]),
            baseline_command=str(experiment.get("baseline_command", "")),
            candidate_command=str(experiment.get("candidate_command", "")),
            metric=str(experiment.get("metric", "")),
            threshold=str(experiment.get("threshold", "")),
            rationale=str(experiment["rationale"]),
        ),
        notification_decision=NotificationDecision(
            notify=bool(notification["notify"]),
            confidence=float(notification["confidence"]),
            reason=str(notification["reason"]),
        ),
        interpretation=str(payload["interpretation"]),
    )


def require_keys(payload: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"OpenAI PI {context} missing required field(s): {', '.join(missing)}")
