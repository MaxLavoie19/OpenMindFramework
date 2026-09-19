import numpy as np
import pytest

from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.model.position_row import PositionRow
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.service.consequence_library_tests import Declare, position, strip_domain
from openmind.rbs.service.reading_cache import ReadingCache
from openmind.rbs.service.rule_compiler import RuleCompiler
from openmind.rbs.service.rule_runner import RuleRunner
from openmind.rbs.service.term_evaluator import TermEvaluator

pytestmark = pytest.mark.log_level("INFO")

#: X threatens to win, valued for X then for O, and the empty strip valued for X.
ROWS = (
    PositionRow(position({1: "X"}, "O"), "X", 1.0),
    PositionRow(position({1: "X"}, "O"), "O", 0.0),
    PositionRow(position({}, "X"), "X", 0.5),
)


def new_evaluator(workers: int = 1) -> TermEvaluator:
    compiler = RuleCompiler()
    runner = RuleRunner(StateNamespaceMapper())
    library = ConsequenceLibraryBuilder().build()
    return TermEvaluator(compiler, runner, library, TaskRunner(workers), ReadingCache(compiler, runner, library, MemoryMeter()))


def test_a_term_reads_the_state_and_the_names_with_me_being_the_row_player(declared: Declare) -> None:
    evaluator = new_evaluator()

    counts = evaluator.column(strip_domain(declared), ROWS, PythonRule("len(cell.where(me))"))
    wins = evaluator.column(strip_domain(declared), ROWS, PythonRule("wins(me)"))
    my_turn = evaluator.column(strip_domain(declared), ROWS, PythonRule("turn == me"))

    assert (counts.tolist(), wins.tolist(), my_turn.tolist()) == ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0])  # type: ignore[union-attr]


def test_a_term_giving_none_is_blank_on_that_row(declared: Declare) -> None:
    column = new_evaluator().column(strip_domain(declared), ROWS, PythonRule("1 if wins(me) else None"))

    assert column is not None and column[0] == 1.0 and np.isnan(column[1:]).all()


@pytest.mark.parametrize("source", ["lamp == 1", "turn", "float('inf')"])
def test_a_term_that_raises_or_gives_no_finite_number_has_no_column(declared: Declare, source: str) -> None:
    assert new_evaluator().column(strip_domain(declared), ROWS, PythonRule(source)) is None


def test_workers_give_the_same_columns(declared: Declare) -> None:
    terms = [PythonRule("wins(me)"), PythonRule("cell[1, 1] == me"), PythonRule("lamp")]

    alone, together = (new_evaluator(workers).columns(strip_domain(declared), ROWS * 3, terms) for workers in (1, 2))

    assert [None if column is None else column.tolist() for column in together] == [
        None if column is None else column.tolist() for column in alone
    ]
    assert together[1].tolist() == [1.0, 0.0, 0.0] * 3  # type: ignore[union-attr]
    assert together[2] is None
