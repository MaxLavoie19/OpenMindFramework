import logging
import os
from pathlib import Path

import pytest

from openmind.parallel.model.call_over_memory import CallOverMemory
from openmind.parallel.model.dropped_call import DroppedCall
from openmind.parallel.model.memory_cap import MemoryCap
from openmind.parallel.model.worker_ended import WorkerEnded
from openmind.parallel.service.task_runner import TaskRunner, _watch_parent

pytestmark = pytest.mark.log_level("INFO")

MB = 1024**2
#: A call, run by exec, that holds 400 MB until its worker is ended.
HOLD = "import time\nballast = bytearray(b'1') * (400 * 1024**2)\ntime.sleep(60)"


def test_one_worker_calls_the_function_here_in_order() -> None:
    calls: list[int] = []

    def double(value: int) -> int:
        calls.append(value)
        return value * 2

    assert TaskRunner(1).map(double, [1, 2, 3]) == [2, 4, 6]
    assert calls == [1, 2, 3]


def test_several_workers_give_the_results_in_order() -> None:
    assert TaskRunner(2).map(pow, [2, 3, 4, 5], [2, 2, 2, 2]) == [4, 9, 16, 25]


def test_workers_log_through_this_process(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("openmind.parallel.worker")

    TaskRunner(2).map(logger.info, ["first", "second", "third"])

    assert sorted(record.getMessage() for record in caplog.records if record.name == "openmind.parallel.worker") == [
        "first",
        "second",
        "third",
    ]


def test_a_failure_in_a_worker_is_raised_here() -> None:
    with pytest.raises(ZeroDivisionError):
        TaskRunner(2).map(divmod, [1, 2], [0, 0])


def test_a_call_ending_its_worker_runs_again_in_a_fresh_worker_then_raises(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)

    with pytest.raises(WorkerEnded, match="the second time with exit code 1"):
        TaskRunner(2).map(os._exit, [1, 1])

    assert any("ended abruptly with exit code 1 during call" in message for message in caplog.messages)
    assert any(message.startswith("Running call ") and "again in a fresh worker" in message for message in caplog.messages)


def test_a_droppable_call_over_the_memory_cap_twice_is_dropped_and_the_other_calls_go_on(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)

    results = TaskRunner(2, MemoryCap(200 * MB, tmp_path, 0.0)).map(exec, [HOLD, "x = 1", "y = 2"], droppable=True)

    dropped = results[0]
    assert results[1:] == [None, None]
    assert isinstance(dropped, DroppedCall) and dropped.index == 0 and dropped.diagnosis is not None
    text = dropped.diagnosis.read_text(encoding="utf-8")
    assert f"stayed over its memory cap of {200 * MB} bytes" in text and "call 0: exec" in text
    assert dropped.diagnosis.with_suffix(".pickle").is_file()
    assert sum("stayed over its memory cap" in message for message in caplog.messages) == 2
    assert any(message.startswith("Dropped call 0 of exec") for message in caplog.messages)


def test_a_call_that_cant_be_dropped_raises_once_over_the_memory_cap_twice(tmp_path: Path) -> None:
    with pytest.raises(CallOverMemory) as raised:
        TaskRunner(2, MemoryCap(200 * MB, tmp_path, 0.0)).map(exec, [HOLD, "x = 1"])

    assert raised.value.index == 0 and raised.value.diagnosis is not None and raised.value.diagnosis.is_file()


@pytest.mark.parametrize("workers", [1, 2])
def test_a_stream_chooses_each_call_s_arguments_when_it_starts_and_sees_every_result_as_it_ends(workers: int) -> None:
    finished: list[int] = []
    seen_when_starting: list[int] = []

    def arguments_for(index: int) -> tuple[int, int]:
        seen_when_starting.append(len(finished))
        return (index + 1, 2)

    def on_result(index: int, result: int) -> None:
        finished.append(index)

    results = TaskRunner(workers).stream(pow, 5, arguments_for, on_result)

    assert results == [1, 4, 9, 16, 25]
    assert sorted(finished) == [0, 1, 2, 3, 4]
    assert seen_when_starting[:workers] == [0] * workers and seen_when_starting[-1] >= 1


def test_a_stream_of_no_call_gives_nothing_and_a_negative_count_raises() -> None:
    assert TaskRunner(2).stream(pow, 0, lambda index: (1, 1), lambda index, result: None) == []
    with pytest.raises(ValueError, match="0 calls or more"):
        TaskRunner(2).stream(pow, -1, lambda index: (1, 1), lambda index, result: None)


def test_a_worker_ends_itself_once_its_parent_is_gone() -> None:
    parents = iter([100, 100, 1])
    ended: list[int] = []
    slept: list[float] = []

    _watch_parent(100, lambda: next(parents), ended.append, slept.append)

    assert (ended, slept) == ([1], [5.0, 5.0])


def test_argument_lists_of_different_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="as many items"):
        TaskRunner(1).map(pow, [1, 2], [1])


def test_fewer_than_one_worker_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1 worker"):
        TaskRunner(0)


def test_split_gives_one_slice_to_one_worker_and_up_to_four_per_worker_otherwise() -> None:
    items = list(range(10))

    assert TaskRunner(1).split(items) == [items]
    assert TaskRunner(2).split(items) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert TaskRunner(8).split(items[:3]) == [[0], [1], [2]]
    assert TaskRunner(2).split([]) == []
