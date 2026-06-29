from pathlib import Path

from labclaw.multimodal_reader import build_user_content, extract_with_gemma, parse_local_fixture


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
