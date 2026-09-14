import logging
import math
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import QueueHandler, QueueListener
from queue import Queue

from openmind.parallel.constant.parallel_constant import SLICES_PER_WORKER, START_METHOD

logger = logging.getLogger(__name__)


class TaskRunner:
    """Runs a function over lists of arguments, in worker processes when it has more than one worker, and gives the
    results in order. Workers start as fresh interpreters, so the function, its arguments and its results travel
    between processes by pickling: the function must be importable, or the method of an object that pickles. Each
    worker sends its log records here, where the root logger's handlers write them, from the root logger's level up.
    With one worker, or a single call, everything runs in this process."""

    def __init__(self, workers: int) -> None:
        if workers < 1:
            raise ValueError(f"A task runner needs at least 1 worker, not {workers}")
        self._workers = workers

    @property
    def workers(self) -> int:
        return self._workers

    def map[R](self, function: Callable[..., R], *arguments: Sequence[object]) -> list[R]:
        """Calls the function with the items at each index of the argument lists, which must be as long as each other."""
        count = len(arguments[0]) if arguments else 0
        if any(len(argument) != count for argument in arguments):
            raise ValueError("Every argument list needs as many items")
        if self._workers == 1 or count <= 1:
            return [function(*items) for items in zip(*arguments, strict=True)]
        workers = min(self._workers, count)
        logger.debug("Running %d calls in %d worker processes", count, workers)
        context = multiprocessing.get_context(START_METHOD)
        queue = context.Queue()
        root = logging.getLogger()
        listener = QueueListener(queue, *root.handlers, respect_handler_level=True)
        listener.start()
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context, initializer=_send_logs, initargs=(queue, root.level)
            ) as pool:
                return list(pool.map(function, *arguments))
        finally:
            listener.stop()

    def split[T](self, items: Sequence[T]) -> list[Sequence[T]]:
        """Contiguous slices of the items, in order: one slice with one worker, otherwise up to SLICES_PER_WORKER slices
        per worker, so that work set up once per slice is set up rarely."""
        if not items:
            return []
        count = 1 if self._workers == 1 else min(len(items), self._workers * SLICES_PER_WORKER)
        size = math.ceil(len(items) / count)
        return [items[start : start + size] for start in range(0, len(items), size)]


def _send_logs(queue: Queue[logging.LogRecord], level: int) -> None:
    """Makes a worker's root logger send every record from the level up to the queue."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(QueueHandler(queue))
    root.setLevel(level)
