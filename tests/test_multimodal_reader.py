from pathlib import Path
import threading
import time

import pytest

import labclaw.multimodal_reader as reader
from labclaw.multimodal_reader import (
    ClaimCard,
    ReaderResult,
    ReaderSource,
    SourceRecord,
    build_user_content,
    format_human_result,
    extract_with_gemma,
    load_json_object,
    parse_local_fixture,
    read_source_record,
    read_source_text,
    read_sources,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


class FakeResponse:
    choices = [
        type(
            "Choice",
            (),
            {
                "message": type(
                    "Message",
                    (),
                    {
                        "content": """{
                          "source": {"title": "Example", "path": "paper.md"},
                          "cards": [{
                            "id": "claim-1",
                            "main_claim": "A scheduler improves benchmark throughput by 1.4x.",
                            "figures": [{
                              "id": "figure-1",
                              "alt_text": "Benchmark bars",
                              "path": "figures/scheduler-bars.png",
                              "caption": "Figure 2. Throughput bars.",
                              "visual_observation": "The scheduler bar is higher than baseline."
                            }],
                            "benchmark_numbers": ["1.4x", "70 tok/s"],
                            "code_hooks": ["python bench.py --scheduler"],
                            "is_testable": true,
                            "evidence_needed": ["run benchmark in VM"]
                          }]
                        }"""
                    },
                )()
            },
        )()
    ]


def test_extracts_claim_card_through_gemma_client(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text(
        """# Example

This repo claims that a scheduler improves benchmark throughput by 1.4x.

![Benchmark bars](figures/scheduler-bars.png)
Figure 2. Throughput bars for baseline and scheduler.

```bash
python bench.py --scheduler
```
""",
        encoding="utf-8",
    )
    client = FakeClient()

    result = extract_with_gemma(source.read_text(encoding="utf-8"), source_path=source, client=client)
    call = client.chat.completions.calls[0]
    card = result.cards[0]

    assert call["model"] == "gpt-4o-mini"
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["temperature"] == 0.2
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["content"][0]["type"] == "text"
    assert card.main_claim == "A scheduler improves benchmark throughput by 1.4x."
    assert card.figures[0].visual_observation == "The scheduler bar is higher than baseline."
    assert card.code_hooks == ["python bench.py --scheduler"]
    assert card.is_testable is True


def test_build_user_content_attaches_supported_local_images_only(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("# Example", encoding="utf-8")
    for index in range(6):
        (tmp_path / f"figure-{index}.png").write_bytes(b"fake-png")
    (tmp_path / "diagram.svg").write_text("<svg />", encoding="utf-8")

    content = "\n".join(
        [f"![Figure {index}](figure-{index}.png)" for index in range(6)]
        + ["![Diagram](diagram.svg)"]
    )
    payload = build_user_content(
        content,
        source_path=source,
        figure_refs=[
            *[{"alt_text": f"Figure {index}", "path": f"figure-{index}.png"} for index in range(6)],
            {"alt_text": "Diagram", "path": "diagram.svg"},
        ],
    )

    assert payload[0]["type"] == "text"
    assert [part["type"] for part in payload[1:]] == ["image_url"] * 5
    assert all(part["image_url"]["url"].startswith("data:image/png;base64,") for part in payload[1:])


def test_build_user_content_skips_oversized_image_and_keeps_later_valid_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "paper.md"
    source.write_text("# Example", encoding="utf-8")
    (tmp_path / "first.png").write_bytes(b"first")
    (tmp_path / "huge.png").write_bytes(b"huge")
    (tmp_path / "last.png").write_bytes(b"last")
    original_stat = Path.stat

    def fake_stat(path: Path):
        stat = original_stat(path)
        if path.name == "huge.png":
            return type("FakeStat", (), {"st_size": 11 * 1024 * 1024})()
        return stat

    monkeypatch.setattr(Path, "stat", fake_stat)

    payload = build_user_content(
        "# Example",
        source_path=source,
        figure_refs=[
            {"alt_text": "First", "path": "first.png"},
            {"alt_text": "Huge", "path": "huge.png"},
            {"alt_text": "Last", "path": "last.png"},
        ],
    )

    image_urls = [part["image_url"]["url"] for part in payload[1:]]
    assert len(image_urls) == 2
    assert image_urls[0].endswith("Zmlyc3Q=")
    assert image_urls[1].endswith("bGFzdA==")


def test_build_user_content_counts_images_explicitly(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("# Example", encoding="utf-8")
    for index in range(6):
        (tmp_path / f"figure-{index}.png").write_bytes(b"fake-png")

    payload = build_user_content(
        "# Example",
        source_path=source,
        figure_refs=[{"alt_text": f"Figure {index}", "path": f"figure-{index}.png"} for index in range(6)],
    )

    assert payload[0]["type"] == "text"
    assert len(payload[1:]) == 5


def test_local_fixture_parser_keeps_offline_smoke_test() -> None:
    result = parse_local_fixture(
        """# Example

This repo claims that a scheduler improves benchmark throughput by 1.4x.

![Benchmark bars](figures/scheduler-bars.png)

```bash
python bench.py --scheduler
```
"""
    )

    card = result.cards[0]

    assert len(card.figures) == 1
    assert "1.4x" in card.benchmark_numbers
    assert card.code_hooks == ["python bench.py --scheduler"]
    assert card.is_testable is True


def test_read_source_text_accepts_in_memory_source_title() -> None:
    result = read_source_text(
        """This repo claims decoding is 1.8x faster.

```bash
python bench.py --decode
```""",
        title="Scout Raw Text",
        use_gemma=False,
    )

    assert result.source == {"title": "Scout Raw Text", "path": None}
    assert result.cards[0].main_claim == "This repo claims decoding is 1.8x faster."
    assert result.cards[0].benchmark_numbers == ["1.8x"]
    assert result.cards[0].code_hooks == ["python bench.py --decode"]


def test_source_record_figures_are_attached_for_gemma(tmp_path: Path) -> None:
    figure = tmp_path / "chart.png"
    figure.write_bytes(b"png-data")
    record = SourceRecord(
        source_id="src-1",
        kind="paper",
        title="Scout Record",
        raw_text="This paper claims kernels improve throughput by 1.4x.",
        figures=[{"alt_text": "Throughput chart", "path": str(figure)}],
    )
    client = FakeClient()

    result = read_source_record(record, client=client)
    payload = client.chat.completions.calls[0]["messages"][1]["content"]

    assert result.source["title"] == "Scout Record"
    assert [part["type"] for part in payload] == ["text", "image_url"]
    assert payload[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_read_sources_preserves_order_and_isolates_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_source_text(content: str, **kwargs) -> ReaderResult:
        if kwargs["title"] == "Broken":
            raise ValueError("bad source")
        return ReaderResult(
            source={"title": kwargs["title"], "path": None},
            cards=[
                ClaimCard(
                    id=f"claim-{kwargs['title'].lower()}",
                    main_claim=content,
                    figures=[],
                    benchmark_numbers=[],
                    code_hooks=[],
                    is_testable=False,
                    evidence_needed=[],
                )
            ],
        )

    monkeypatch.setattr(reader, "read_source_text", fake_read_source_text)

    results = read_sources(
        [
            ReaderSource(source_id="a", title="First", content="first"),
            ReaderSource(source_id="b", title="Broken", content="broken"),
            ReaderSource(source_id="c", title="Third", content="third"),
        ],
        max_workers=2,
        use_gemma=False,
    )

    assert [result.source_id for result in results] == ["a", "b", "c"]
    assert [result.ok for result in results] == [True, False, True]
    assert results[0].card_count == 1
    assert results[1].error == "ValueError: bad source"
    assert results[2].result is not None


def test_read_sources_respects_configured_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = threading.Lock()
    active = 0
    max_seen = 0

    def slow_read_source_text(content: str, **kwargs) -> ReaderResult:
        nonlocal active, max_seen
        with lock:
            active += 1
            max_seen = max(max_seen, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return ReaderResult(
            source={"title": kwargs["title"], "path": None},
            cards=[],
        )

    monkeypatch.setattr(reader, "read_source_text", slow_read_source_text)

    results = read_sources(
        [ReaderSource(source_id=str(index), title=f"Source {index}", content="claim") for index in range(5)],
        max_workers=2,
        use_gemma=False,
    )

    assert len(results) == 5
    assert max_seen <= 2
    assert all(result.ok for result in results)
    assert all(result.elapsed_ms > 0 for result in results)


def test_load_json_object_rejects_none_content() -> None:
    with pytest.raises(ValueError, match="no message content"):
        load_json_object(None)


def test_load_json_object_wraps_malformed_json() -> None:
    with pytest.raises(ValueError, match="malformed JSON"):
        load_json_object("{not json")


def test_human_output_handles_empty_cards() -> None:
    output = format_human_result(ReaderResult(source={"title": "Empty", "path": None}, cards=[]))

    assert output == "# Empty\nNo claims extracted."


def test_format_human_result_prints_all_cards() -> None:
    result = ReaderResult(
        source={"title": "Multi", "path": None},
        cards=[
            ClaimCard(
                id="claim-1",
                main_claim="First claim.",
                figures=[],
                benchmark_numbers=["1.2x"],
                code_hooks=["python first.py"],
                is_testable=True,
                evidence_needed=[],
            ),
            ClaimCard(
                id="claim-2",
                main_claim="Second claim.",
                figures=[],
                benchmark_numbers=["2.0x"],
                code_hooks=["python second.py"],
                is_testable=True,
                evidence_needed=[],
            ),
        ],
    )

    output = format_human_result(result)

    assert "## Claim 1" in output
    assert "First claim." in output
    assert "## Claim 2" in output
    assert "Second claim." in output
