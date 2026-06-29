"""Group raw sources into durable ML research topics.

Turns the source firehose into a small set of stable clusters so LabClaw
behaves like a lab with memory, not a feed reader. Cluster IDs are stable
(derived from the topic slug), so the same topic seen across many heartbeats
maps to the same cluster instead of spawning duplicates. Memory is a plain
JSON file so it survives process restarts.

Assignment is pluggable. The default KeywordLabeler is deterministic and
offline (good for tests and a no-API baseline); a real embedding/LLM labeler
can be dropped in behind the same interface.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

UNCLUSTERED_ID = "unclustered"

# topic_slug -> keywords that signal it.
DEFAULT_TOPICS = {
    "small-language-model-training": [
        "small language model", "slm", "tiny model", "pretraining",
        "training recipe", "from scratch", "nanogpt", "tokenizer",
    ],
    "inference-speed-kernels": [
        "inference", "kernel", "throughput", "latency", "tokens/sec",
        "cuda", "flash attention", "quantization", "kv cache", "wall-clock",
    ],
    "agentic-coding": [
        "agent", "coding agent", "swe-bench", "tool use", "autonomous",
        "code generation", "pass@1", "repository",
    ],
    "distillation-synthetic-data": [
        "distillation", "synthetic data", "teacher model", "student model",
        "self-instruct", "data generation",
    ],
    "eval-harnesses": [
        "eval", "benchmark", "harness", "leaderboard", "lm-eval",
        "evaluation suite", "test set",
    ],
    "long-context-methods": [
        "long context", "context window", "rope", "position encoding",
        "sliding window", "needle in a haystack", "128k",
    ],
    "rag-and-memory": [
        "rag", "retrieval augmented", "vector store", "embedding retrieval",
        "memory", "reranker", "knowledge base",
    ],
}


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "topic"


@dataclass
class LabelResult:
    topic: Optional[str]  # slug, or None when ambiguous
    score: float
    scores: dict


class KeywordLabeler:
    """Deterministic offline labeler: keyword-overlap scoring per topic."""

    def __init__(self, topics: Optional[dict] = None, min_score: float = 0.34) -> None:
        self.topics = topics or DEFAULT_TOPICS
        self.min_score = min_score

    def label(self, text: str) -> LabelResult:
        text_l = (text or "").lower()
        scores: dict = {}
        for topic, keywords in self.topics.items():
            hits = sum(1 for kw in keywords if kw in text_l)
            # Normalize by a small constant so a couple of strong hits is enough,
            # but a single incidental hit stays below threshold (ambiguous).
            scores[topic] = hits / 3.0
        if not scores:
            return LabelResult(None, 0.0, {})
        best_topic = max(scores, key=scores.get)
        best = scores[best_topic]
        if best < self.min_score:
            return LabelResult(None, best, scores)
        return LabelResult(best_topic, min(best, 1.0), scores)


@dataclass
class TopicCluster:
    cluster_id: str
    topic_name: str
    source_ids: list = field(default_factory=list)
    claim_cards: list = field(default_factory=list)
    experiments: list = field(default_factory=list)
    results: list = field(default_factory=list)
    trend: str = "unknown"
    open_questions: list = field(default_factory=list)
    next_experiment: Optional[str] = None
    keywords: list = field(default_factory=list)
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "topic_name": self.topic_name,
            "source_ids": self.source_ids,
            "claim_cards": self.claim_cards,
            "experiments": self.experiments,
            "results": self.results,
            "trend": self.trend,
            "open_questions": self.open_questions,
            "next_experiment": self.next_experiment,
            "keywords": self.keywords,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopicCluster":
        return cls(
            cluster_id=d["cluster_id"],
            topic_name=d["topic_name"],
            source_ids=list(d.get("source_ids", [])),
            claim_cards=list(d.get("claim_cards", [])),
            experiments=list(d.get("experiments", [])),
            results=list(d.get("results", [])),
            trend=d.get("trend", "unknown"),
            open_questions=list(d.get("open_questions", [])),
            next_experiment=d.get("next_experiment"),
            keywords=list(d.get("keywords", [])),
            updated_at=d.get("updated_at"),
        )


@dataclass
class Assignment:
    cluster_id: str
    is_new_cluster: bool
    is_duplicate: bool
    ambiguous: bool
    novelty: float
    topic_score: float


class ClusterStore:
    """Durable JSON-backed cluster memory that survives heartbeats."""

    def __init__(self, path, labeler: Optional[KeywordLabeler] = None) -> None:
        self.path = Path(path)
        self.labeler = labeler or KeywordLabeler()
        self.clusters: dict = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.clusters = {
                c["cluster_id"]: TopicCluster.from_dict(c) for c in data.get("clusters", [])
            }
        else:
            self.clusters = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"clusters": [c.to_dict() for c in self.clusters.values()]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _text_for(record) -> str:
        # Accepts a SourceRecord or a plain dict.
        get = record.get if isinstance(record, dict) else lambda k, d="": getattr(record, k, d)
        parts = [get("title", ""), get("raw_text", "")]
        meta = get("metadata", {}) or {}
        if isinstance(meta, dict):
            parts.append(" ".join(str(t) for t in meta.get("topics", [])))
        return " ".join(p for p in parts if p)

    @staticmethod
    def _source_id_of(record) -> str:
        return record["source_id"] if isinstance(record, dict) else record.source_id

    def assign(self, record, *, novelty_floor: float = 0.0) -> Assignment:
        """Assign a source to a cluster, creating one if the topic is new.

        - Stable cluster_id (topic slug) => same topic never duplicates.
        - Repeated source_id => no-op, flagged is_duplicate.
        - Ambiguous text => parked in the 'unclustered' cluster for review.
        - novelty = 1.0 for a brand-new cluster, else 1 - topic_score (capped).
        """
        source_id = self._source_id_of(record)
        result = self.labeler.label(self._text_for(record))

        ambiguous = result.topic is None
        cluster_id = UNCLUSTERED_ID if ambiguous else result.topic
        topic_name = "Unclustered / needs review" if ambiguous else result.topic.replace("-", " ")

        existing = self.clusters.get(cluster_id)
        is_new = existing is None
        if is_new:
            existing = TopicCluster(
                cluster_id=cluster_id,
                topic_name=topic_name,
                keywords=self.labeler.topics.get(cluster_id, []) if not ambiguous else [],
            )
            self.clusters[cluster_id] = existing

        is_duplicate = source_id in existing.source_ids
        if not is_duplicate:
            existing.source_ids.append(source_id)
        existing.updated_at = _now()

        novelty = 1.0 if is_new else max(novelty_floor, 1.0 - result.score)
        return Assignment(
            cluster_id=cluster_id,
            is_new_cluster=is_new,
            is_duplicate=is_duplicate,
            ambiguous=ambiguous,
            novelty=round(novelty, 3),
            topic_score=round(result.score, 3),
        )

    def digest(self) -> list:
        """Cluster-level summary for the Gemini PI."""
        return [
            {
                "cluster_id": c.cluster_id,
                "topic_name": c.topic_name,
                "num_sources": len(c.source_ids),
                "trend": c.trend,
                "next_experiment": c.next_experiment,
            }
            for c in self.clusters.values()
        ]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
