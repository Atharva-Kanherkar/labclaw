"""Top-level `labclaw` CLI. Currently exposes the scout subcommand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from labclaw.clustering import ClusterStore
from labclaw.daemon import main as daemon_main
from labclaw.figures import FigureStore
from labclaw.sources import (
    ArxivScout,
    Fetcher,
    GitHubScout,
    SeenStore,
    run_scouts,
)


def _scout(args) -> int:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher()
    seen = SeenStore(data_dir / "seen.json")
    figure_store = FigureStore(data_dir / "figures", fetcher)

    scouts = [
        ArxivScout(fetcher, max_results=args.max),
        GitHubScout(fetcher, max_results=args.max),
    ]
    records = run_scouts(scouts, seen_store=seen, figure_store=figure_store)
    seen.save()

    out = data_dir / "sources.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict()) + "\n")

    clusters = ClusterStore(data_dir / "clusters.json")
    for r in records:
        clusters.assign(r)
    clusters.save()

    print(
        f"scout: {len(records)} new source(s) from {len(scouts)} scout(s); "
        f"{len(clusters.clusters)} cluster(s). Written to {out}."
    )
    return 0


def _daemon(args) -> int:
    daemon_main(getattr(args, "daemon_args", []))
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="labclaw", description="LabClaw CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scout = sub.add_parser("scout", help="Fetch fresh sources once.")
    p_scout.add_argument("--once", action="store_true", help="Run a single scout pass.")
    p_scout.add_argument("--max", type=int, default=25, help="Max results per scout.")
    p_scout.add_argument("--data-dir", default="labclaw_data", help="Where to store state.")
    p_scout.set_defaults(func=_scout)

    p_daemon = sub.add_parser("daemon", help="Run the heartbeat daemon.")
    p_daemon.add_argument("daemon_args", nargs=argparse.REMAINDER, help="Daemon flags such as --once")
    p_daemon.set_defaults(func=_daemon)

    args = parser.parse_args(argv)
    try:
        raise SystemExit(args.func(args))
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
