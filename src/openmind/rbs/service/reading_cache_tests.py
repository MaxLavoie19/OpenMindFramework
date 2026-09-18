import pytest

from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.consequence_library_tests import Declare, position, strip_domain
from openmind.rbs.service.reading_cache import ReadingCache
from openmind.rbs.service.rule_compiler import RuleCompiler
from openmind.rbs.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

pytestmark = pytest.mark.log_level("INFO")

MINE = "{view}.cell[i] == me"
THEIRS = "{view}.cell[i] == other"
ROW = PositionRow(position({1: "X", 2: "O"}, "X"), "X", 0.0)


def new_cache() -> ReadingCache:
    compiler, runner = RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper()))
    return ReadingCache(compiler, runner, ConsequenceLibraryBuilder().build(), MemoryMeter())


def test_a_reading_gives_a_value_at_every_index_of_the_base(declared: Declare) -> None:
    cache = new_cache()

    mine = cache.values(strip_domain(declared), ROW, "cell", MINE)
    theirs = cache.values(strip_domain(declared), ROW, "cell", THEIRS)

    # The strip's cells are (1, 1) to (1, 4): X holds the first and O the second.
    assert mine == (1.0, 0.0, 0.0, 0.0)
    assert theirs == (0.0, 1.0, 0.0, 0.0)


def test_a_reading_is_read_once_for_a_position_and_player(declared: Declare) -> None:
    cache = new_cache()
    other = PositionRow(ROW.state, "O", 0.0)

    first = cache.values(strip_domain(declared), ROW, "cell", MINE)

    assert cache.values(strip_domain(declared), ROW, "cell", MINE) is first
    assert cache.values(strip_domain(declared), other, "cell", MINE) == (0.0, 1.0, 0.0, 0.0)
    assert cache.memory_entries() == 2


def test_a_reading_that_cant_be_read_gives_nothing_and_clearing_forgets_what_was_read(declared: Declare) -> None:
    cache = new_cache()

    assert cache.values(strip_domain(declared), ROW, "cell", "{view}.nothing[i]") == ()
    cache.values(strip_domain(declared), ROW, "cell", MINE)
    cache.clear()

    assert cache.memory_entries() == 0
