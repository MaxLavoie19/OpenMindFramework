import logging

from openmind.debug.mapper.session_json_mapper import SessionJsonMapper
from openmind.debug.model.breakpoint import Breakpoint
from openmind.debug.model.debug_session import DebugSession
from openmind.debug.model.target import Target
from openmind.debug.model.verbosity import Verbosity
from openmind.rule.model.python_rule import PythonRule


def test_a_session_goes_to_json_and_comes_back_equal() -> None:
    session = DebugSession(
        "en passant",
        (Verbosity(logging.INFO), Verbosity(logging.DEBUG, Target("openmind.mcts", "chess", "evaluation", (("topic", "pawns"),)))),
        (Breakpoint("en passant", "evaluation", PythonRule("en_passant is not None")),),
    )
    mapper = SessionJsonMapper()

    assert mapper.from_text(mapper.to_text(session)) == session
