import logging
from pathlib import Path

import pytest

from openmind.debug.constant.debug_constant import CONTINUE, LOGGED, RUN, STEP
from openmind.debug.model.breakpoint import Breakpoint
from openmind.debug.model.debug_session import DebugSession
from openmind.debug.model.pause import Pause
from openmind.debug.model.target import Target
from openmind.debug.model.verbosity import Verbosity
from openmind.debug.model.warning import Warning
from openmind.debug.service.debugger import Debugger
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.grid import Grid
from openmind.world.model.state import State

logger = logging.getLogger("openmind.search")

POSITION = State.of(cell=Grid.filled((3, 3), None).placed((2, 2), "X"), turn="O")


class Answering:
    """A pauser answering each pause in turn, keeping the pauses."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.pauses: list[Pause] = []

    def __call__(self, pause: Pause) -> str:
        self.pauses.append(pause)
        return self.answers.pop(0) if self.answers else CONTINUE


@pytest.fixture
def debugger() -> Debugger:
    made = Debugger()
    yield made
    made.stop()


def test_without_breakpoints_or_targets_frames_are_no_ops(debugger: Debugger) -> None:
    debugger.start(DebugSession("quiet"))

    with debugger.frame("search") as opened:
        assert opened is None and debugger.stack() == ()


def test_a_breakpoint_stops_where_its_condition_holds_with_both_stacks(debugger: Debugger) -> None:
    pauser = Answering()
    centre = Breakpoint("centre taken", "evaluation", PythonRule("cell[2, 2] == 'X' and turn == 'O'"))
    debugger.start(DebugSession("centre", breakpoints=(centre,)), pauser=pauser)

    with debugger.frame("search", context="tictactoe"):
        with debugger.frame("evaluation", state=State.of(cell=Grid.filled((3, 3), None), turn="O")):
            pass
        with debugger.frame("evaluation", state=POSITION, details={"player": "O"}):
            pass

    (pause,) = pauser.pauses
    assert pause.reason == "breakpoint centre taken"
    assert [frame.kind for frame in pause.stack] == ["search", "evaluation"]
    assert pause.stack[-1].python.endswith(".py:" + pause.stack[-1].python.rsplit(":", 1)[1])
    assert "debugger_tests.py" in pause.python_stack


def test_stepping_stops_at_the_next_frame_and_an_interrupt_at_the_next_one_opened(debugger: Debugger) -> None:
    pauser = Answering(STEP)
    debugger.start(DebugSession("stepping", breakpoints=(Breakpoint("search", "search"),)), pauser=pauser)

    with debugger.frame("search"):
        with debugger.frame("evaluation"):
            pass
        with debugger.frame("evaluation"):
            pass
    debugger.interrupt()
    with debugger.frame("task"):
        pass

    assert [pause.reason for pause in pauser.pauses] == ["breakpoint search", "stepped", "interrupted"]


def test_targeted_verbosity_keeps_debug_lines_only_inside_the_frames_it_targets(debugger: Debugger, tmp_path: Path) -> None:
    session = DebugSession(
        "evaluations",
        (Verbosity(logging.INFO), Verbosity(logging.DEBUG, Target(frame="evaluation"))),
        log_file=tmp_path / "run.log",
    )
    debugger.start(session)

    logger.debug("outside any frame")
    with debugger.frame("evaluation"):
        logger.debug("inside an evaluation")
    logger.info("an info line")
    debugger.stop()

    text = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "inside an evaluation" in text and "an info line" in text and "outside any frame" not in text


def test_every_omf_warning_is_kept_and_written_to_the_knowledge_base(debugger: Debugger, tmp_path: Path) -> None:
    knowledge = create_knowledge_base("tests", tmp_path)
    debugger.start(DebugSession("warnings"), knowledge)

    logger.warning("a frozen rule can't explain e.p.")
    debugger.warn(Warning("conflicting rules", "two rules disagree", None, ("rule-1", "rule-2")))

    assert [(warning.kind, warning.ids) for warning in debugger.warnings()] == [
        (LOGGED, ("openmind.search",)),
        ("conflicting rules", ("rule-1", "rule-2")),
    ]
    kept = knowledge.experiences(tags=(("keyword", "warning"),))
    assert [experience.value for experience in kept] == ["a frozen rule can't explain e.p.", "two rules disagree"]
    assert kept[1].source.rests_on == ("rule-1", "rule-2")


def test_clocks_freeze_while_paused_unless_the_session_lets_them_run(debugger: Debugger) -> None:
    debugger.start(DebugSession("frozen", breakpoints=(Breakpoint("any", "search"),)), pauser=lambda pause: CONTINUE)
    with debugger.frame("search"):
        pass

    assert debugger.frozen_seconds() >= 0.0 and debugger.measurement_kept(float("inf"))
    assert not debugger.measurement_kept(0.0)

    debugger.start(DebugSession("running", clocks_while_paused=RUN, keep_paused_measurements=True))
    assert debugger.frozen_seconds() == 0.0 and debugger.measurement_kept(0.0)
