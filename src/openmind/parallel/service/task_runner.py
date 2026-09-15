import logging
import math
import multiprocessing
import os
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from logging.handlers import QueueHandler
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path

from openmind.parallel.constant.parallel_constant import (
    MEMORY_EXIT_CODE,
    PARENT_CHECK_SECONDS,
    SLICES_PER_WORKER,
    START_METHOD,
    STOP_SECONDS,
)
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.model.call_over_memory import CallOverMemory
from openmind.parallel.model.dropped_call import DroppedCall
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.model.worker_ended import WorkerEnded

logger = logging.getLogger(__name__)

_MISSING = object()


class TaskRunner:
    """Runs a function over lists of arguments, in worker processes when it has more than one worker, and gives the
    results in order. Workers start as fresh interpreters, so the function, its arguments and its results travel
    between processes by pickling: the function must be importable, or the method of an object that pickles. Each
    worker sends its log records here, where the root logger's handlers write them, from the root logger's level up.
    With one worker, or a single call, everything runs in this process.

    Each worker talks to this process through a pipe of its own, so a worker that ends abruptly loses only the call it
    was running: that call runs again in a fresh worker. A call that ends a fresh worker too raises `WorkerEnded`, or,
    when its worker stayed over the memory cap both times, `CallOverMemory`; a droppable call gives a `DroppedCall` in
    its place instead, and the other calls go on. Under a memory cap, each worker's memory guard clears its caches above
    the cap and ends the worker with a diagnosis once clearing isn't enough (see `MemoryGuard`). After each call, a
    worker collects its garbage and hands freed memory back to the system. A worker whose parent is gone, killed or
    ended, ends itself instead of lingering."""

    def __init__(self, workers: int, memory_cap: MemoryCap | None = None) -> None:
        if workers < 1:
            raise ValueError(f"A task runner needs at least 1 worker, not {workers}")
        self._workers = workers
        self._memory_cap = memory_cap

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def memory_cap(self) -> MemoryCap | None:
        return self._memory_cap

    def map[R](self, function: Callable[..., R], *arguments: Sequence[object], droppable: bool = False) -> list[R]:
        """Calls the function with the items at each index of the argument lists, which must be as long as each other.
        With `droppable`, a call that stayed over the memory cap in a fresh worker too gives a `DroppedCall` instead of
        its result."""
        count = len(arguments[0]) if arguments else 0
        if any(len(argument) != count for argument in arguments):
            raise ValueError("Every argument list needs as many items")
        if self._workers == 1 or count <= 1:
            return [function(*items) for items in zip(*arguments, strict=True)]
        calls = list(zip(*arguments, strict=True))
        return _Run(function, calls, min(self._workers, count), self._memory_cap, droppable).results()  # type: ignore[return-value]

    def split[T](self, items: Sequence[T]) -> list[Sequence[T]]:
        """Contiguous slices of the items, in order: one slice with one worker, otherwise up to SLICES_PER_WORKER slices
        per worker, so that work set up once per slice is set up rarely."""
        if not items:
            return []
        count = 1 if self._workers == 1 else min(len(items), self._workers * SLICES_PER_WORKER)
        size = math.ceil(len(items) / count)
        return [items[start : start + size] for start in range(0, len(items), size)]


@dataclass(slots=True)
class _Worker:
    """A worker process, the pipe to it, the call it runs, and the diagnosis it sent before ending over its memory cap."""

    process: BaseProcess
    connection: Connection
    call: int | None = None
    over: Path | None = None
    gone: bool = False


class _Run:
    """One map over worker processes: hands the calls out one at a time, collects results and log records, and replaces
    the workers that end."""

    def __init__(
        self,
        function: Callable[..., object],
        calls: list[tuple[object, ...]],
        workers: int,
        memory_cap: MemoryCap | None,
        droppable: bool,
    ) -> None:
        self._function = function
        self._calls = calls
        self._workers = workers
        self._memory_cap = memory_cap
        self._droppable = droppable
        self._context = multiprocessing.get_context(START_METHOD)
        self._results: list[object] = [_MISSING] * len(calls)
        self._waiting: deque[int] = deque(range(len(calls)))
        self._failures: dict[int, int] = {}
        self._live: list[_Worker] = []
        self._done = 0

    def results(self) -> list[object]:
        logger.debug("Running %d calls in %d worker processes", len(self._calls), self._workers)
        finished = False
        try:
            for _ in range(self._workers):
                self._assign(self._start())
            while self._done < len(self._calls):
                ready = wait([worker.connection for worker in self._live] + [worker.process.sentinel for worker in self._live])
                for worker in list(self._live):
                    ended = worker.process.sentinel in ready
                    if worker.connection in ready or ended:
                        self._receive(worker)
                    if worker.gone or ended:
                        self._end(worker)
            finished = True
            return self._results
        finally:
            self._stop(finished)

    def _start(self) -> _Worker:
        here, there = self._context.Pipe()
        process = self._context.Process(
            target=_work, args=(there, logging.getLogger().level, os.getpid(), self._memory_cap), name="openmind worker"
        )
        process.start()
        there.close()
        worker = _Worker(process, here)
        self._live.append(worker)
        return worker

    def _assign(self, worker: _Worker) -> None:
        if not self._waiting:
            return
        index = self._waiting.popleft()
        worker.call, worker.over = index, None
        try:
            worker.connection.send((index, self._function, self._calls[index]))
        except OSError:
            worker.gone = True

    def _receive(self, worker: _Worker) -> None:
        """Reads everything the worker sent so far."""
        while not worker.gone:
            try:
                if not worker.connection.poll():
                    return
                message = worker.connection.recv()
            except (EOFError, OSError):
                worker.gone = True
                return
            except Exception as error:  # noqa: BLE001 - a message cut short by a worker ending unpickles as anything.
                logger.warning("Could not read what worker %s sent: %r", worker.process.pid, error)
                worker.gone = True
                return
            kind = message[0]
            if kind == "log":
                _handle(message[1])
            elif kind == "done":
                _, index, value = message
                self._results[index] = value
                self._done += 1
                worker.call = None
                self._assign(worker)
            elif kind == "error":
                _, index, error, text = message
                error.add_note(f"Raised in worker {worker.process.pid} during call {index}:\n{text}")
                raise error
            elif kind == "over":
                worker.over = message[2]

    def _end(self, worker: _Worker) -> None:
        """Forgets a worker that ended, runs its call again or gives up on it, and starts a fresh worker for the calls
        left."""
        worker.process.join(STOP_SECONDS)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join()
        worker.connection.close()
        self._live.remove(worker)
        if worker.call is not None:
            self._lost(worker)
        if self._waiting:
            self._assign(self._start())

    def _lost(self, worker: _Worker) -> None:
        index, code, pid = worker.call, worker.process.exitcode, worker.process.pid
        assert index is not None
        name = getattr(self._function, "__qualname__", repr(self._function))
        over = worker.over is not None or code == MEMORY_EXIT_CODE
        self._failures[index] = self._failures.get(index, 0) + 1
        if over:
            logger.warning(
                "Worker %s stayed over its memory cap of %s bytes during call %d of %s and ended; diagnosis %s",
                pid,
                None if self._memory_cap is None else self._memory_cap.worker_bytes,
                index,
                name,
                worker.over,
            )
        else:
            logger.warning("Worker %s ended abruptly with exit code %s during call %d of %s", pid, code, index, name)
        if self._failures[index] == 1:
            logger.warning("Running call %d of %s again in a fresh worker", index, name)
            self._waiting.appendleft(index)
        elif not over:
            raise WorkerEnded(f"Call {index} of {name} ended its worker twice, the second time with exit code {code}")
        elif not self._droppable:
            raise CallOverMemory(index, worker.over)
        else:
            logger.warning(
                "Dropped call %d of %s: it stayed over the memory cap in a fresh worker too; diagnosis %s",
                index,
                name,
                worker.over,
            )
            self._results[index] = DroppedCall(index, worker.over)
            self._done += 1

    def _stop(self, finished: bool) -> None:
        """Asks every worker to end once the calls are done; terminates them when a call failed."""
        for worker in self._live:
            if finished:
                try:
                    worker.connection.send(None)
                except OSError:
                    pass
            else:
                worker.process.terminate()
        for worker in self._live:
            worker.process.join(STOP_SECONDS)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join()
            worker.connection.close()
        self._live.clear()


class _Sender:
    """What a worker's log handler puts records into: the pipe to the process that started it."""

    def __init__(self, send: Callable[[object], None]) -> None:
        self._send = send

    def put_nowait(self, record: logging.LogRecord) -> None:
        self._send(("log", record))


def _handle(record: logging.LogRecord) -> None:
    for handler in logging.getLogger().handlers:
        if record.levelno >= handler.level:
            handler.handle(record)


def _work(connection: Connection, level: int, parent: int, memory_cap: MemoryCap | None) -> None:
    """A worker: sends its logs through its pipe, watches its parent and, under a cap, its memory, then runs calls until
    told to end. Everything it writes to the pipe goes through one lock, and a worker ending over its memory cap keeps
    that lock, so nothing is left half written."""
    lock = threading.Lock()

    def send(message: object) -> None:
        with lock:
            connection.send(message)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(QueueHandler(_Sender(send)))  # type: ignore[arg-type]
    root.setLevel(level)
    threading.Thread(target=_watch_parent, args=(parent,), name="parent watch", daemon=True).start()
    guard = process_memory_guard()
    if memory_cap is not None:

        def report_over(index: int | None, diagnosis: Path) -> None:
            send(("over", index, diagnosis))
            lock.acquire()

        guard.watch(memory_cap, report_over)
    while True:
        try:
            task = connection.recv()
        except EOFError:
            return
        if task is None:
            return
        index, function, arguments = task
        guard.calling(index, function, arguments)
        try:
            message: tuple[object, ...] = ("done", index, function(*arguments))
        except Exception as error:  # noqa: BLE001 - every failure is raised again in the process that started the worker.
            message = ("error", index, error, traceback.format_exc())
        guard.called()
        try:
            send(message)
        except Exception as error:  # noqa: BLE001 - a result or error that doesn't pickle.
            send(("error", index, RuntimeError(f"Call {index} gave what can't be sent back: {error!r}"), traceback.format_exc()))
        del message, task, function, arguments
        guard.release()


def _watch_parent(
    parent: int,
    parent_now: Callable[[], int] = os.getppid,
    end: Callable[[int], object] = os._exit,
    sleep: Callable[[float], object] = time.sleep,
) -> None:
    """Ends this worker once its parent is gone: a worker whose parent was killed is adopted by another process, so its
    parent's id changes. Checked every PARENT_CHECK_SECONDS."""
    while parent_now() == parent:
        sleep(PARENT_CHECK_SECONDS)
    end(1)
