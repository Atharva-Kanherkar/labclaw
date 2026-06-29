"""Choose one testable ML/code claim and emit a VM experiment spec."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from labclaw.multimodal_reader import ClaimCard, FigureReference, ReaderResult, result_from_dict

RUNNABLE_PREFIXES = (
    "python ",
    "python3 ",
    "pytest",
    "uv ",
    "pip install",
    "git clone",
    "make",
    "cargo ",
    "go test",
)


@dataclass(frozen=True)
class ClaimScore:
    total: int
    runnable_hooks: int
    benchmark_numbers: int
    figure_evidence: int
    evidence_needs: int


@dataclass(frozen=True)
class ExperimentSpec:
    claim_id: str
    main_claim: str
    score: ClaimScore
    setup_commands: list[str]
    run_commands: list[str]
    success_signals: list[str]
    figure_paths: list[str]
    benchmark_numbers: list[str]
    rationale: str


def pick_claim(result: ReaderResult) -> ExperimentSpec | None:
    ranked: list[tuple[int, ClaimScore, ClaimCard]] = []
    for index, card in enumerate(result.cards):
        score = score_claim(card)
        if has_reproduction_path(card, score):
            ranked.append((index, score, card))

    if not ranked:
        return None

    index, score, card = max(ranked, key=lambda item: (item[1].total, -item[0]))
    return build_experiment_spec(card, score)


def score_claim(card: ClaimCard) -> ClaimScore:
    runnable_hooks = sum(1 for hook in card.code_hooks if is_runnable_hook(hook))
    benchmark_numbers = len(card.benchmark_numbers)
    figure_evidence = sum(1 for figure in card.figures if figure.path or figure.visual_observation)
    evidence_needs = len(card.evidence_needed)
    total = (
        runnable_hooks * 4
        + benchmark_numbers * 3
        + figure_evidence * 2
        + evidence_needs
        + (2 if card.is_testable else 0)
    )
    return ClaimScore(
        total=total,
        runnable_hooks=runnable_hooks,
        benchmark_numbers=benchmark_numbers,
        figure_evidence=figure_evidence,
        evidence_needs=evidence_needs,
    )


def has_reproduction_path(card: ClaimCard, score: ClaimScore) -> bool:
    return score.runnable_hooks > 0 and (score.benchmark_numbers > 0 or score.figure_evidence > 0)


def build_experiment_spec(card: ClaimCard, score: ClaimScore) -> ExperimentSpec:
    runnable = [hook for hook in card.code_hooks if is_runnable_hook(hook)]
    setup_commands = [hook for hook in runnable if hook.startswith(("git clone", "pip install", "uv pip install"))]
    run_commands = [hook for hook in runnable if hook not in setup_commands]
    success_signals = [
        *[f"compare metric: {number}" for number in card.benchmark_numbers],
        *[f"inspect figure: {figure.path}" for figure in card.figures if figure.path],
        *card.evidence_needed,
    ]
    return ExperimentSpec(
        claim_id=card.id,
        main_claim=card.main_claim,
        score=score,
        setup_commands=setup_commands,
        run_commands=run_commands,
        success_signals=list(dict.fromkeys(success_signals)),
        figure_paths=[figure.path for figure in card.figures if figure.path],
        benchmark_numbers=card.benchmark_numbers,
        rationale=(
            "Selected because it has runnable commands plus "
            "benchmark or figure evidence that can be checked in a VM."
        ),
    )


def is_runnable_hook(command: str) -> bool:
    stripped = command.strip()
    return any(stripped.startswith(prefix) for prefix in RUNNABLE_PREFIXES)


def load_reader_result(path: Path) -> ReaderResult:
    return result_from_dict(json.loads(path.read_text(encoding="utf-8")))


def spec_to_dict(spec: ExperimentSpec | None) -> dict[str, Any]:
    if spec is None:
        return {"selected": None, "reason": "No claim had a runnable reproduction path."}
    return {"selected": asdict(spec)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick one LabClaw claim to reproduce/refute.")
    parser.add_argument("reader_json", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    spec = pick_claim(load_reader_result(args.reader_json))
    if args.json:
        print(json.dumps(spec_to_dict(spec), indent=2))
        return

    if spec is None:
        print("No claim had a runnable reproduction path.")
        return

    print(f"Claim: {spec.main_claim}")
    print(f"Score: {spec.score.total}")
    print(f"Run commands: {len(spec.run_commands)}")
    print(f"Success signals: {len(spec.success_signals)}")


if __name__ == "__main__":
    main()
