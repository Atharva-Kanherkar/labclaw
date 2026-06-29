import json
from pathlib import Path

from labclaw.speed_demo import (
    format_timing_table,
    load_fixture_batch,
    progress_json_lines,
    run_speed_demo,
)


def test_load_fixture_batch_repeats_sources_with_stable_ids(tmp_path: Path) -> None:
    fixture = tmp_path / "claim.md"
    fixture.write_text(
        """# Kernel Claim

This repo claims a fused kernel improves throughput by 1.5x.

```bash
python bench.py --kernel fused
```""",
        encoding="utf-8",
    )

    sources = load_fixture_batch([fixture], repeat=3)

    assert [source.source_id for source in sources] == ["claim-1", "claim-2", "claim-3"]
    assert all(source.title == "Kernel Claim" for source in sources)
    assert all(source.metadata["cluster_id"] == "demo-reader-swarm" for source in sources)


def test_run_speed_demo_reports_lanes_progress_and_cluster(tmp_path: Path) -> None:
    fixture = tmp_path / "claim.md"
    fixture.write_text(
        """# Reader Claim

This repo claims batching improves reader throughput by 1.2x.

```bash
python bench.py --batch
```""",
        encoding="utf-8",
    )
    sources = load_fixture_batch([fixture], repeat=2)
    events = []

    report = run_speed_demo(
        sources,
        fast_workers=2,
        baseline_delay_ms=1,
        progress=events.append,
    )

    assert report.selected_cluster == "demo-reader-swarm"
    assert [lane.name for lane in report.lanes] == ["cerebras-gemma-swarm", "simulated-slower-baseline"]
    assert [lane.sources_completed for lane in report.lanes] == [2, 2]
    assert all(lane.claim_cards_produced == 2 for lane in report.lanes)
    assert any(event.event == "lane_started" for event in events)
    assert sum(event.event == "source_completed" for event in events) == 4
    assert all(result.ok for result in report.fast_results)
    assert all(result.ok for result in report.baseline_results)


def test_format_timing_table_includes_demo_metrics(tmp_path: Path) -> None:
    fixture = tmp_path / "claim.md"
    fixture.write_text(
        """# Timing Claim

This repo claims decoding is 1.1x faster.

```bash
python bench.py --decode
```""",
        encoding="utf-8",
    )

    report = run_speed_demo(load_fixture_batch([fixture], repeat=1), baseline_delay_ms=1)
    table = format_timing_table(report)

    assert "Selected cluster: demo-reader-swarm" in table
    assert "cerebras-gemma-swarm" in table
    assert "simulated-slower-baseline" in table
    assert "Est tok/s" in table
    assert "Claim cards" in table


def test_progress_json_lines_are_machine_readable(tmp_path: Path) -> None:
    fixture = tmp_path / "claim.md"
    fixture.write_text(
        """# JSON Claim

This repo claims caching improves throughput by 1.3x.

```bash
python bench.py --cache
```""",
        encoding="utf-8",
    )

    report = run_speed_demo(load_fixture_batch([fixture], repeat=1), baseline_delay_ms=1)
    lines = progress_json_lines(report.progress_events).splitlines()

    assert lines
    decoded = [json.loads(line) for line in lines]
    assert decoded[0]["event"] == "lane_started"
    assert all("elapsed_ms" in event for event in decoded)
