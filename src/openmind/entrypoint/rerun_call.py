import argparse
import logging
import pickle
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

from openmind.parallel.constant.parallel_constant import DEFAULT_RERUN_LINES

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Runs a call a worker ended on for its memory cap again, alone in this process, while tracemalloc traces its
    allocations, then prints and logs the lines that hold the most memory once the call is over, with the peak."""
    parser = argparse.ArgumentParser(
        prog="openmind-rerun-call",
        description="Run again, alone, the call a worker ended on for its memory cap, and show where it holds memory.",
    )
    parser.add_argument("call", type=Path, help="the pickled call a memory diagnosis names, or the diagnosis itself")
    parser.add_argument(
        "--lines",
        type=int,
        default=DEFAULT_RERUN_LINES,
        help=f"source lines shown, those holding the most memory first (default: {DEFAULT_RERUN_LINES})",
    )
    parser.add_argument(
        "--log-directory", default="data/log/rerun-call", help="where logs are saved (default: data/log/rerun-call)"
    )
    arguments = parser.parse_args(argv)
    path = arguments.call if arguments.call.suffix == ".pickle" else arguments.call.with_suffix(".pickle")
    if not path.is_file():
        parser.error(f"no pickled call at {path}")

    directory = Path(arguments.log_directory)
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)-5s %(name)s %(message)s"))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        with path.open("rb") as file:
            function, call_arguments = pickle.load(file)
        name = getattr(function, "__qualname__", repr(function))
        logger.info("Running %s again from %s under tracemalloc", name, path)
        tracemalloc.start()
        started = time.monotonic()
        outcome = "ended"
        result = None
        try:
            result = function(*call_arguments)
        except Exception:
            outcome = "raised"
            logger.exception("The call raised")
        snapshot = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        seconds = time.monotonic() - started
        header = ("bytes", "blocks", "line")
        rows = [
            (str(statistic.size), str(statistic.count), str(statistic.traceback))
            for statistic in snapshot.statistics("lineno")[: arguments.lines]
        ]
        summary = (
            f"The call {outcome} after {seconds:.1f} seconds; traced memory peaked at {peak} bytes and holds {current} "
            f"bytes with the call's objects still alive"
        )
        print(summary)
        logger.info(summary)
        widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]
        for row in (header, *rows):
            line = "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
            print(line)
            logger.info(line)
        del result
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
