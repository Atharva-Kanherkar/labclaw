"""Resolve user-supplied inputs (arxiv, twitter, files) into verifiable sources."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from labclaw.sources import ARXIV_API, SourceRecord

ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+")
CLAIM_HINT = re.compile(r"\b(\d+(?:\.\d+)?x|\d+(?:\.\d+)?%|\d+\s*tok/s)\b", re.IGNORECASE)


def resolve_input(raw: str, *, fetcher=None) -> SourceRecord:
    """Turn a Telegram message, URL, arXiv id, or file path into a SourceRecord."""
    text = raw.strip()
    if not text:
        raise ValueError("Empty input.")

    path = Path(text)
    if path.exists() and path.is_file():
        return _from_file(path)

    arxiv_match = ARXIV_ID.search(text)
    if arxiv_match:
        return _from_arxiv(arxiv_match.group(1), fetcher=fetcher)

    url_match = URL_PATTERN.search(text)
    if url_match:
        url = url_match.group(0).rstrip(").,")
        if "arxiv.org" in url:
            arxiv = ARXIV_ID.search(url)
            if arxiv:
                return _from_arxiv(arxiv.group(1), fetcher=fetcher)
        return _from_url(url, text, fetcher=fetcher)

    if text.lower().startswith("arxiv:"):
        return _from_arxiv(text.split(":", 1)[1].strip(), fetcher=fetcher)

    return _from_text(text)


def _from_file(path: Path) -> SourceRecord:
    content = path.read_text(encoding="utf-8")
    title = next((line.removeprefix("# ").strip() for line in content.splitlines() if line.startswith("# ")), path.stem)
    return SourceRecord(
        source_id=f"file:{path.stem}",
        kind="paper",
        title=title,
        url=path.as_uri(),
        raw_text=content,
        metadata={"input_kind": "file", "path": str(path)},
    )


def _from_arxiv(arxiv_id: str, *, fetcher=None) -> SourceRecord:
    query = f"id_list={urllib.parse.quote(arxiv_id)}"
    url = f"{ARXIV_API}?{query}"
    xml_text = _fetch_text(url, fetcher=fetcher)
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"arXiv entry not found: {arxiv_id}")
    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
    link = next(
        (node.get("href") for node in entry.findall("atom:link", ns) if node.get("rel") == "alternate"),
        f"https://arxiv.org/abs/{arxiv_id}",
    )
    published = entry.findtext("atom:published", default=None, namespaces=ns)
    body = f"# {title}\n\n{summary}\n"
    return SourceRecord(
        source_id=f"arxiv:{arxiv_id}",
        kind="paper",
        title=title,
        url=link,
        published_at=published,
        raw_text=body,
        metadata={"input_kind": "arxiv", "arxiv_id": arxiv_id},
    )


def _from_url(url: str, surrounding_text: str, *, fetcher=None) -> SourceRecord:
    html = _fetch_text(url, fetcher=fetcher)
    title_match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else url
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    if surrounding_text and surrounding_text != url:
        text = f"{surrounding_text}\n\n{text[:8000]}"
    return SourceRecord(
        source_id=f"url:{urllib.parse.quote(url, safe='')[:48]}",
        kind="blog",
        title=title,
        url=url,
        raw_text=text[:12000],
        metadata={"input_kind": "url"},
    )


def _from_text(text: str) -> SourceRecord:
    hints = CLAIM_HINT.findall(text)
    title = text.splitlines()[0][:120] if text.splitlines() else "User claim"
    return SourceRecord(
        source_id="text:user-claim",
        kind="blog",
        title=title,
        url="",
        raw_text=text,
        metadata={"input_kind": "text", "claim_hints": hints},
    )


def _fetch_text(url: str, *, fetcher=None) -> str:
    if fetcher is not None:
        if hasattr(fetcher, "get_text"):
            return fetcher.get_text(url)
        return fetcher(url).decode("utf-8", errors="replace")
    req = urllib.request.Request(url, headers={"User-Agent": "labclaw-verify/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")
