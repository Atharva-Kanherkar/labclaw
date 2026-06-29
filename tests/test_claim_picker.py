from typing import Optional

from labclaw.claim_picker import build_experiment_spec, pick_claim, score_claim
from labclaw.multimodal_reader import ClaimCard, FigureReference, ReaderResult


def claim(
    claim_id: str,
    *,
    hooks: list[str],
    numbers: list[str],
    figures: Optional[list[FigureReference]] = None,
    testable: bool = True,
) -> ClaimCard:
    return ClaimCard(
        id=claim_id,
        main_claim=f"{claim_id} improves throughput.",
        figures=figures or [],
        benchmark_numbers=numbers,
        code_hooks=hooks,
        is_testable=testable,
        evidence_needed=["run benchmark"] if testable else [],
    )


def test_pick_claim_prefers_runnable_measured_visual_claim() -> None:
    visual = FigureReference(
        id="fig-1",
        alt_text="curve",
        path="figures/curve.png",
        caption="Loss curve.",
        visual_observation="The optimized curve converges faster.",
    )
    weak = claim("claim-1", hooks=["python bench.py"], numbers=[])
    strong = claim("claim-2", hooks=["python bench.py"], numbers=["1.3x"], figures=[visual])

    spec = pick_claim(ReaderResult(source={"title": "Example", "path": None}, cards=[weak, strong]))

    assert spec is not None
    assert spec.claim_id == "claim-2"
    assert "compare metric: 1.3x" in spec.success_signals
    assert "inspect figure: figures/curve.png" in spec.success_signals


def test_pick_claim_returns_none_without_reproduction_path() -> None:
    result = ReaderResult(
        source={"title": "Example", "path": None},
        cards=[claim("claim-1", hooks=[], numbers=["1.3x"])],
    )

    assert pick_claim(result) is None


def test_build_experiment_spec_uses_first_safe_command_sequence() -> None:
    card = claim(
        "claim-1",
        hooks=[
            "git clone https://github.com/example/repo",
            "pip install -r requirements.txt",
            "python bench.py --baseline",
        ],
        numbers=["70 tok/s"],
    )

    spec = build_experiment_spec(card, score_claim(card))

    assert spec.setup_commands == [
        "git clone https://github.com/example/repo",
        "pip install -r requirements.txt",
    ]
    assert spec.run_commands == ["python bench.py --baseline"]
    assert "compare metric: 70 tok/s" in spec.success_signals


def test_pick_claim_tie_breaks_by_original_order() -> None:
    first = claim("claim-1", hooks=["python first.py"], numbers=["1.0x"])
    second = claim("claim-2", hooks=["python second.py"], numbers=["1.0x"])

    spec = pick_claim(ReaderResult(source={"title": "Example", "path": None}, cards=[first, second]))

    assert spec is not None
    assert spec.claim_id == "claim-1"
