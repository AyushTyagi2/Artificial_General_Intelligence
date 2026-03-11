"""Entrypoint for running the digital baby prototype."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from digital_baby.engine.event_loop import BabyEventLoop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the curiosity-driven digital baby agent.")
    parser.add_argument(
        "--world",
        default=str(Path(__file__).parent / "world" / "knowledge_pages"),
        help="Path to folder containing knowledge page JSON files.",
    )
    parser.add_argument(
        "--memory",
        default=str(Path(__file__).parent / "world" / "memory_store.json"),
        help="Path to JSON memory persistence file.",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between ticks.")
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Optional finite number of ticks for testing/demo. Default runs forever.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args()

    loop = BabyEventLoop(world_path=args.world, memory_path=args.memory, tick_sleep_seconds=args.sleep)
    loop.run(max_ticks=args.ticks)


if __name__ == "__main__":
    main()
