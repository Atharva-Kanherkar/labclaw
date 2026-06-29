from pathlib import Path

import pytest

from labclaw.multimodal_reader import (
    ClaimCard,
    ReaderResult,
    build_user_content,
    format_human_result,
    extract_with_gemma,
    load_json_object,
    parse_local_fixture,
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

    assert call["model"] == "gemma-4-31b"
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
