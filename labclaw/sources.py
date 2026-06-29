"""Automated source scouts.

Answers "where do papers/repos come from?" automatically. Manual URL/file is
fallback only. Each scout discovers fresh research and emits SourceRecords of a
single agreed shape so downstream lanes (reader swarm, clustering) can consume
fixtures without coupling to scout internals.

All network access goes through an injectable Fetcher, so tests run entirely on
recorded fixtures with no live calls. Figures are downloaded and stored locally
as PNG/JPEG via FigureStore (issue #16's reader cannot use remote image URLs).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from labclaw.figures import Figure, FigureStore

ARXIV_API = "http://export.arxiv.org/api/query"
GITHUB_SEARCH_API = "https://api.github.com/search/repositories"
DEFAULT_ARXIV_CATEGORIES = ["cs.LG", "cs.AI", "cs.CL"]
VALID_KINDS = {"paper", "repo", "model", "benchmark", "blog"}

_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)")
_HTML_IMG = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.IGNORECASE)


@dataclass
class SourceRecord:
    """The single shape every scout publishes."""

    source_id: str
    kind: str
    title: str
    url: str
    published_at: Optional[str] = None
    raw_text: str = ""
    figures: list = field(default_factory=list)  # list[Figure]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "title": self.title,
            "url": self.url,
            "published_at": self.published_at,
            "raw_text": self.raw_text,
            "figures": [f.to_dict() for f in self.figures],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SourceRecord":
        return cls(
            source_id=d["source_id"],
            kind=d["kind"],
            title=d.get("title", ""),
            url=d.get("url", ""),
            published_at=d.get("published_at"),
            raw_text=d.get("raw_text", ""),
            figures=[Figure.from_dict(f) for f in d.get("figures", [])],
            metadata=d.get("metadata", {}),
        )


# --------------------------------------------------------------------------- #
# Fetchers (injectable so tests stay offline)
# --------------------------------------------------------------------------- #


class Fetcher:
    """Default fetcher backed by urllib. Subclass/replace for tests."""

    def __init__(self, headers: Optional[dict] = None, timeout: int = 30) -> None:
        self.headers = headers or {"User-Agent": "labclaw-scout/0.1"}
        self.timeout = timeout

    def get_bytes(self, url: str, headers: Optional[dict] = None) -> bytes:
        req = urllib.request.Request(url, headers={**self.headers, **(headers or {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def get_text(self, url: str, headers: Optional[dict] = None) -> str:
        return self.get_bytes(url, headers).decode("utf-8", errors="replace")

    def __call__(self, url: str) -> bytes:  # FigureStore expects a bytes callable
        return self.get_bytes(url)


class MappingFetcher(Fetcher):
    """Offline fetcher: serves recorded fixtures keyed by URL (exact or prefix).

    Values may be bytes or a path to a fixture file. Unknown URLs raise, so a
    test can never accidentally hit the network.
    """

    def __init__(self, mapping: dict) -> None:
        super().__init__()
        self.mapping = mapping
        self.requested: list = []

    def _resolve(self, url: str) -> bytes:
        self.requested.append(url)
        val = self.mapping.get(url)
        if val is None:
            for key, v in self.mapping.items():
                if url.startswith(key):
                    val = v
                    break
        if val is None:
            raise KeyError(f"No recorded fixture for URL: {url}")
        if isinstance(val, (bytes, bytearray)):
            return bytes(val)
        return Path(val).read_bytes()

    def get_bytes(self, url: str, headers: Optional[dict] = None) -> bytes:
        return self._resolve(url)

    def __call__(self, url: str) -> bytes:
        return self._resolve(url)


# --------------------------------------------------------------------------- #
# Dedupe
# --------------------------------------------------------------------------- #


class SeenStore:
    """Durable set of already-ingested source_ids (JSON file)."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.seen: set = set()
        if self.path.exists():
            self.seen = set(json.loads(self.path.read_text(encoding="utf-8")))

    def __contains__(self, source_id: str) -> bool:
        return source_id in self.seen

    def add(self, source_id: str) -> None:
        self.seen.add(source_id)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(self.seen)), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Scouts
# --------------------------------------------------------------------------- #


class Scout(ABC):
    name = "scout"

    @abstractmethod
    def discover(self) -> list:
        """Return a list of SourceRecord (figures still as remote URLs)."""


def extract_figure_urls(text: str, base_url: str = "") -> list:
    """Pull image URLs (markdown + html) out of README/HTML text."""
    urls = []
    for m in _MD_IMAGE.finditer(text or ""):
        urls.append((m.group(1).strip(), m.group(2).strip()))
    for m in _HTML_IMG.finditer(text or ""):
        urls.append(("", m.group(1).strip()))
    resolved = []
    for alt, u in urls:
        if u.startswith(("http://", "https://", "data:")):
            if not u.startswith("data:"):
                resolved.append((alt, u))
        elif base_url:
            resolved.append((alt, urllib.parse.urljoin(base_url, u)))
    return resolved


class ArxivScout(Scout):
    name = "arxiv"

    def __init__(
        self,
        fetcher: Fetcher,
        categories: Optional[list] = None,
        queries: Optional[list] = None,
        max_results: int = 25,
    ) -> None:
        self.fetcher = fetcher
        self.categories = categories or DEFAULT_ARXIV_CATEGORIES
        self.queries = queries or []
        self.max_results = max_results

    def query_url(self) -> str:
        cat_q = " OR ".join(f"cat:{c}" for c in self.categories)
        terms = [f"({cat_q})"] if cat_q else []
        terms += [f"all:{q}" for q in self.queries]
        search = " OR ".join(terms) if terms else "all:machine learning"
        params = {
            "search_query": search,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(self.max_results),
        }
        return f"{ARXIV_API}?{urllib.parse.urlencode(params)}"

    def discover(self) -> list:
        xml = self.fetcher.get_text(self.query_url())
        return self._parse(xml)

    @staticmethod
    def _parse(xml: str) -> list:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml)
        records = []
        for entry in root.findall("a:entry", ns):
            raw_id = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
            arxiv_id = raw_id.rsplit("/", 1)[-1]
            arxiv_id_nover = re.sub(r"v\d+$", "", arxiv_id)
            title = " ".join((entry.findtext("a:title", "", ns) or "").split())
            summary = (entry.findtext("a:summary", "", ns) or "").strip()
            published = entry.findtext("a:published", None, ns)
            cats = [
                c.attrib.get("term")
                for c in entry.findall("a:category", ns)
                if c.attrib.get("term")
            ]
            records.append(
                SourceRecord(
                    source_id=f"arxiv:{arxiv_id_nover}",
                    kind="paper",
                    title=title,
                    url=raw_id,
                    published_at=published,
                    raw_text=summary,
                    metadata={"arxiv_id": arxiv_id, "categories": cats, "topics": cats},
                )
            )
        return records


class GitHubScout(Scout):
    name = "github"

    def __init__(
        self,
        fetcher: Fetcher,
        query: str = "machine learning benchmark",
        max_results: int = 25,
        token: Optional[str] = None,
        fetch_readme: bool = True,
    ) -> None:
        self.fetcher = fetcher
        self.query = query
        self.max_results = max_results
        self.token = token
        self.fetch_readme = fetch_readme

    def search_url(self) -> str:
        params = {
            "q": self.query,
            "sort": "updated",
            "order": "desc",
            "per_page": str(self.max_results),
        }
        return f"{GITHUB_SEARCH_API}?{urllib.parse.urlencode(params)}"

    def _headers(self) -> dict:
        h = {"Accept": "application/vnd.github+json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def discover(self) -> list:
        body = self.fetcher.get_text(self.search_url(), headers=self._headers())
        data = json.loads(body)
        records = []
        for item in data.get("items", []):
            full_name = item.get("full_name", "")
            raw_text = item.get("description", "") or ""
            if self.fetch_readme and item.get("default_branch"):
                readme = self._try_readme(full_name, item["default_branch"])
                if readme:
                    raw_text = f"{raw_text}\n\n{readme}".strip()
            records.append(
                SourceRecord(
                    source_id=f"github:{full_name}",
                    kind="repo",
                    title=full_name,
                    url=item.get("html_url", ""),
                    published_at=item.get("pushed_at") or item.get("created_at"),
                    raw_text=raw_text,
                    metadata={
                        "stars": item.get("stargazers_count"),
                        "topics": item.get("topics", []),
                        "default_branch": item.get("default_branch"),
                    },
                )
            )
        return records

    def _try_readme(self, full_name: str, branch: str) -> str:
        url = f"https://raw.githubusercontent.com/{full_name}/{branch}/README.md"
        try:
            return self.fetcher.get_text(url)
        except Exception:
            return ""


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_scouts(
    scouts: list,
    *,
    seen_store: SeenStore,
    figure_store: Optional[FigureStore] = None,
    figure_fetcher: Optional[object] = None,
    max_figures_per_source: int = 5,
) -> list:
    """Discover, dedupe, fetch+store figures locally, and mark seen.

    Returns only the NEW SourceRecords (already-seen ids are skipped). Figures
    are downloaded and stored as local PNG/JPEG; remote URLs that can't be
    transcoded are recorded under metadata['skipped_figures'].
    """
    new_records = []
    for scout in scouts:
        for record in scout.discover():
            if record.source_id in seen_store:
                continue
            seen_store.add(record.source_id)
            if figure_store is not None:
                _attach_figures(record, figure_store, max_figures_per_source)
            new_records.append(record)
    return new_records


def _attach_figures(record, figure_store, limit) -> None:
    skipped = []
    candidates = extract_figure_urls(record.raw_text, base_url=record.url)
    for alt, url in candidates[:limit]:
        fig = figure_store.store(url, alt_text=alt)
        if fig is not None:
            record.figures.append(fig)
        else:
            skipped.append(url)
    if skipped:
        record.metadata["skipped_figures"] = skipped
