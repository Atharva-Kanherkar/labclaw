from pathlib import Path

from labclaw.pipeline import LabPipeline
from labclaw.reproduce_loop import run_reproduce_loop
from labclaw.verify import resolve_input


def test_resolve_input_from_file(tmp_path: Path) -> None:
    source = tmp_path / "claim.md"
    source.write_text("# Demo\n\n42 tok/s baseline and 55 tok/s optimized\n", encoding="utf-8")
    record = resolve_input(str(source))
    assert record.source_id.startswith("file:")
    assert "42 tok/s" in record.raw_text


def test_resolve_input_from_text() -> None:
    record = resolve_input("New kernel gives 2.3x speedup on H100")
    assert record.metadata["input_kind"] == "text"
    assert record.metadata["claim_hints"]


def test_reproduce_loop_finds_improvement() -> None:
    journal = run_reproduce_loop(claim_id="claim-1", baseline=42.0, candidate=55.0, max_attempts=3)
    assert journal.best() is not None
    assert journal.best().kept is True


def test_verify_pipeline_from_sample(tmp_path: Path) -> None:
    pipeline = LabPipeline(tmp_path / "data", fixture_mode=True)
    sample = Path(__file__).resolve().parents[1] / "samples" / "tiny-ml-claim.md"
    source = resolve_input(str(sample))
    result = pipeline.run(source=source)
    assert result.reportable is True
    assert result.critic_verdict["verdict"] == "reproduced"
