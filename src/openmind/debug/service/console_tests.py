from openmind.debug.constant.debug_constant import STEP
from openmind.debug.model.frame import Frame
from openmind.debug.model.pause import Pause
from openmind.debug.service.console import Console
from openmind.world.model.state import State


def test_the_console_shows_the_stacks_and_the_state_then_resumes_as_told() -> None:
    commands = iter(["stack", "state", "step"])
    written: list[str] = []
    pause = Pause(
        "breakpoint e.p.",
        (Frame("search", "chess", python="tree_search.py:70"), Frame("evaluation", state=State.of(turn="white"), python="rule_based_system.py:230")),
        "  File tree_search.py, line 70\n",
        123,
    )

    answer = Console(read=lambda prompt: next(commands), write=written.append)(pause)

    assert answer == STEP
    assert written[0] == "Paused in process 123 (breakpoint e.p.): evaluation at rule_based_system.py:230"
    assert "1. search in chess — tree_search.py:70" in written[1] and "Python stack:" in written[1]
    assert written[2] == "turn = 'white'"
