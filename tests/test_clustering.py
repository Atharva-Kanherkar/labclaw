from labclaw.clustering import (
    UNCLUSTERED_ID,
    ClusterStore,
    KeywordLabeler,
    TopicCluster,
    slugify,
)
from labclaw.sources import SourceRecord


def rec(source_id, title, text="", topics=None):
    return SourceRecord(
        source_id=source_id,
        kind="paper",
        title=title,
        url=f"http://x/{source_id}",
        raw_text=text,
        metadata={"topics": topics or []},
    )


def test_slugify():
    assert slugify("Small Language Model Training") == "small-language-model-training"


def test_labeler_picks_topic_and_flags_ambiguous():
    labeler = KeywordLabeler()
    slm = labeler.label("a small language model pretraining recipe from scratch")
    assert slm.topic == "small-language-model-training"
    assert slm.score >= labeler.min_score

    ambiguous = labeler.label("a general note about something unrelated")
    assert ambiguous.topic is None


def test_assign_new_topic_creates_cluster(tmp_path):
    store = ClusterStore(tmp_path / "clusters.json")
    a = store.assign(rec("arxiv:1", "Long context method", "long context window rope 128k"))
    assert a.is_new_cluster is True
    assert a.cluster_id == "long-context-methods"
    assert a.novelty == 1.0
    assert "arxiv:1" in store.clusters[a.cluster_id].source_ids


def test_existing_topic_no_duplicate_cluster(tmp_path):
    store = ClusterStore(tmp_path / "clusters.json")
    store.assign(rec("arxiv:1", "Eval harness", "benchmark eval harness leaderboard"))
    a2 = store.assign(rec("arxiv:2", "Another eval", "lm-eval benchmark harness test set"))
    assert a2.is_new_cluster is False
    assert a2.cluster_id == "eval-harnesses"
    # one cluster, two sources -- no duplicate cluster
    assert len(store.clusters) == 1
    assert store.clusters["eval-harnesses"].source_ids == ["arxiv:1", "arxiv:2"]
    assert a2.novelty < 1.0


def test_duplicate_source_is_flagged_not_readded(tmp_path):
    store = ClusterStore(tmp_path / "clusters.json")
    r = rec("arxiv:1", "RAG memory", "retrieval augmented generation vector store memory")
    store.assign(r)
    a2 = store.assign(r)
    assert a2.is_duplicate is True
    assert store.clusters["rag-and-memory"].source_ids == ["arxiv:1"]


def test_ambiguous_source_goes_to_unclustered(tmp_path):
    store = ClusterStore(tmp_path / "clusters.json")
    a = store.assign(rec("arxiv:9", "Misc", "an unrelated general note"))
    assert a.ambiguous is True
    assert a.cluster_id == UNCLUSTERED_ID


def test_memory_survives_reload(tmp_path):
    path = tmp_path / "clusters.json"
    store = ClusterStore(path)
    store.assign(rec("arxiv:1", "Kernels", "inference kernel throughput cuda quantization"))
    store.save()

    reloaded = ClusterStore(path)  # fresh instance, same file = survives heartbeat
    assert "inference-speed-kernels" in reloaded.clusters
    assert "arxiv:1" in reloaded.clusters["inference-speed-kernels"].source_ids


def test_cluster_roundtrip():
    c = TopicCluster(cluster_id="x", topic_name="X", source_ids=["a"], trend="up")
    assert TopicCluster.from_dict(c.to_dict()) == c
