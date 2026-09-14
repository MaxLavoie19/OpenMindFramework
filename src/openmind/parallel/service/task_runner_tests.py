import logging

import pytest

from openmind.parallel.service.task_runner import TaskRunner

pytestmark = pytest.mark.log_level("INFO")


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
