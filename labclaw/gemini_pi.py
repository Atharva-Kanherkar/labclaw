"""Backward-compatible shim — LabClaw PI now uses OpenAI."""

from labclaw.openai_pi import (  # noqa: F401
    DEFAULT_PI_MODEL,
    PIInputs,
    PI_PROMPT_VERSION,
    PI_SCHEMA_VERSION,
    ClusterPriority,
    ExperimentProposal,
    NotificationDecision,
    OpenAIAPIClient as GeminiAPIClient,
    OpenAIPI as GeminiPI,
    PIDecision,
    PIDecisionClient,
    SearchPlan,
    build_pi_messages,
    decision_from_payload,
    require_keys,
)
from labclaw.openai_client import load_json_object  # noqa: F401
