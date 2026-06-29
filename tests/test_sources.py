import base64
import json
from pathlib import Path

import pytest

from labclaw.figures import FigureStore
from labclaw.sources import (
    ARXIV_API,
    GITHUB_SEARCH_API,
    ArxivScout,
    GitHubScout,
    MappingFetcher,
    SeenStore,
    SourceRecord,
    extract_figure_urls,
    run_scouts,
)

FIXTURES = Path(__file__).parent / "fixtures"
PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# --------------------------------------------------------------------------- #
# SourceRecord shape
# --------------------------------------------------------------------------- #


def test_source_record_roundtrip():
    r = SourceRecord(
        source_id="arxiv:1", kind="paper", title="T", url="u",
        published_at="2024", raw_text="body", metadata={"a": 1},
    )
    assert SourceRecord.from_dict(r.to_dict()) == r


# --------------------------------------------------------------------------- #
# arXiv scout
# --------------------------------------------------------------------------- #


def test_arxiv_scout_parses_fixture():
    fetcher = MappingFetcher({ARXIV_API: str(FIXTURES / "arxiv_atom.xml")})
    scout = ArxivScout(fetcher, max_results=5)
    records = scout.discover()
    assert len(records) == 2
    first = records[0]
    assert first.source_id == "arxiv:2406.01234"  # version stripped
    assert first.kind == "paper"
    assert "Small Language Model" in first.title
    assert first.metadata["categories"] == ["cs.LG", "cs.CL"]
    assert first.published_at == "2024-06-02T09:00:00Z"


def test_arxiv_query_url_includes_categories():
    fetcher = MappingFetcher({ARXIV_API: b"<feed xmlns='http://www.w3.org/2005/Atom'></feed>"})
    scout = ArxivScout(fetcher, categories=["cs.LG"], queries=["distillation"])
    from urllib.parse import unquote
    url = unquote(scout.query_url())
    assert "cat:cs.LG" in url
    assert "distillation" in url


# --------------------------------------------------------------------------- #
# GitHub scout
# --------------------------------------------------------------------------- #


def test_github_scout_parses_fixture():
    fetcher = MappingFetcher({GITHUB_SEARCH_API: str(FIXTURES / "github_search.json")})
    scout = GitHubScout(fetcher, fetch_readme=False)
    records = scout.discover()
    assert [r.source_id for r in records] == [
        "github:acme/fast-eval-harness",
        "github:acme/rag-memory-lib",
    ]
    assert records[0].kind == "repo"
    assert records[0].metadata["stars"] == 321


def test_github_scout_fetches_readme():
    readme_url = "https://raw.githubusercontent.com/acme/fast-eval-harness/main/README.md"
    fetcher = MappingFetcher({
        GITHUB_SEARCH_API: str(FIXTURES / "github_search.json"),
        readme_url: b"# Fast Eval\nA benchmark harness.",
        "https://raw.githubusercontent.com/acme/rag-memory-lib/main/README.md": b"# RAG",
    })
    scout = GitHubScout(fetcher, fetch_readme=True)
    records = scout.discover()
    assert "Fast Eval" in records[0].raw_text


# --------------------------------------------------------------------------- #
# Figure URL extraction
# --------------------------------------------------------------------------- #


def test_extract_figure_urls_markdown_and_html():
    text = (
        "![loss curve](https://x/loss.png)\n"
        "<img src='https://x/arch.svg'>\n"
        "![rel](figs/local.png)\n"
        "![data uri](data:image/png;base64,AAAA)"
    )
    urls = extract_figure_urls(text, base_url="https://repo/blob/main/README.md")
    found = {u for _, u in urls}
    assert "https://x/loss.png" in found
    assert "https://x/arch.svg" in found
    assert "https://repo/blob/main/figs/local.png" in found  # resolved relative
    assert not any(u.startswith("data:") for u in found)  # data URIs dropped


# --------------------------------------------------------------------------- #
# Dedupe + pipeline
# --------------------------------------------------------------------------- #


def test_seen_store_persists(tmp_path):
    p = tmp_path / "seen.json"
    s = SeenStore(p)
    s.add("arxiv:1")
    s.save()
    assert "arxiv:1" in SeenStore(p)


def test_run_scouts_dedupes_across_runs(tmp_path):
    fetcher = MappingFetcher({ARXIV_API: str(FIXTURES / "arxiv_atom.xml")})
    seen = SeenStore(tmp_path / "seen.json")
    scout = ArxivScout(fetcher, max_results=5)

    first = run_scouts([scout], seen_store=seen)
    assert len(first) == 2  # both new

    second = run_scouts([scout], seen_store=seen)
    assert second == []  # all already seen -> no re-ingestion


def test_run_scouts_stores_figures_locally(tmp_path):
    readme_url = "https://raw.githubusercontent.com/acme/fast-eval-harness/main/README.md"
    rag_readme = "https://raw.githubusercontent.com/acme/rag-memory-lib/main/README.md"
    fetcher = MappingFetcher({
        GITHUB_SEARCH_API: str(FIXTURES / "github_search.json"),
        readme_url: b"# Fast Eval\n![loss](https://cdn/loss.png)\n",
        rag_readme: b"# RAG\n",
        "https://cdn/loss.png": PNG_1x1,
    })
    seen = SeenStore(tmp_path / "seen.json")
    figure_store = FigureStore(tmp_path / "figures", fetcher)
    scout = GitHubScout(fetcher, fetch_readme=True)

    records = run_scouts([scout], seen_store=seen, figure_store=figure_store)
    eval_repo = next(r for r in records if r.source_id == "github:acme/fast-eval-harness")
    assert len(eval_repo.figures) == 1
    fig = eval_repo.figures[0]
    assert fig.path.endswith(".png")
    assert Path(fig.path).exists()  # downloaded + stored locally, not a URL
    assert fig.source_url == "https://cdn/loss.png"
