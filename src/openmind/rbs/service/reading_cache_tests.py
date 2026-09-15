import pytest

from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.consequence_library_tests import position, strip_domain
from openmind.rbs.service.reading_cache import ReadingCache
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

pytestmark = pytest.mark.log_level("INFO")

MINE = "{view}.cell[i] == me"
PARITY = "{view}.cell.parity(i)"
ROW = PositionRow(position({1: "X", 2: "O"}, "X"), "X", 0.0)


def new_cache() -> ReadingCache:
    compiler, runner = RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper()))
    return ReadingCache(compiler, runner, ConsequenceLibraryBuilder().build(), MemoryMeter())


def test_a_reading_gives_a_value_at_every_index_of_the_base() -> None:
    cache = new_cache()

    mine = cache.values(strip_domain(), ROW, "cell", MINE)
    parity = cache.values(strip_domain(), ROW, "cell", PARITY)

    # The strip's cells are (1, 1) to (1, 4): X holds the first, and the parity of 1 + col alternates.
    assert mine == (1.0, 0.0, 0.0, 0.0)
    assert parity == (0.0, 1.0, 0.0, 1.0)


def test_a_reading_is_read_once_for_a_position_and_player() -> None:
    cache = new_cache()
    other = PositionRow(ROW.state, "O", 0.0)

    first = cache.values(strip_domain(), ROW, "cell", MINE)

    assert cache.values(strip_domain(), ROW, "cell", MINE) is first
    assert cache.values(strip_domain(), other, "cell", MINE) == (0.0, 1.0, 0.0, 0.0)
    assert cache.memory_entries() == 2


def test_a_reading_that_cant_be_read_gives_nothing_and_clearing_forgets_what_was_read() -> None:
    cache = new_cache()

    assert cache.values(strip_domain(), ROW, "cell", "{view}.nothing[i]") == ()
    cache.values(strip_domain(), ROW, "cell", MINE)
    cache.clear()

    assert cache.memory_entries() == 0
