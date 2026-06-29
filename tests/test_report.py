import pytest

from labclaw.report import build_report


def test_build_report_for_reportable_run() -> None:
    report = build_report(
        run_id="run-demo",
        cluster_topic="inference speed / kernels",
        source={"title": "Tiny Optimizer", "url": "https://example.com/paper"},
        claim={"main_claim": "Cache-aware batching improves throughput."},
        metric_result={"metric": "tokens_per_second", "baseline": 42.0, "candidate": 55.0},
        critic_verdict={
            "verdict": "reproduced",
            "confidence": 1.0,
            "reportable": True,
            "metric_delta": {
                "metric": "tokens_per_second",
                "baseline": 42.0,
                "candidate": 55.0,
                "delta": 13.0,
            },
            "blocking_objections": [],
        },
    )

    assert report.reportable is True
    assert report.metric_delta == 13.0
    assert "Cache-aware" in report.telegram_ping()


def test_build_report_skips_telegram_when_not_reportable() -> None:
    report = build_report(
        run_id="run-demo",
        cluster_topic="eval harnesses",
        source={"title": "Null result", "url": "https://example.com"},
        claim={"main_claim": "No improvement."},
        metric_result={"metric": "accuracy", "baseline": 0.5, "candidate": 0.51},
        critic_verdict={
            "verdict": "inconclusive",
            "confidence": 0.6,
            "reportable": False,
            "metric_delta": {"metric": "accuracy", "baseline": 0.5, "candidate": 0.51, "delta": 0.01},
            "blocking_objections": ["metric delta did not clear threshold"],
        },
    )

    assert report.telegram_ping() == ""
