import pickle
from collections.abc import Callable

from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.reading_cache import ReadingCache
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner

type Game = Callable[[str], RuleBasedGame]


def caching() -> ReadingCache:
    """Wired as the value generator's builder wires one, in this process."""
    compiler, runner = RuleCompiler(), RuleRunner(StateNamespaceMapper())
    return ReadingCache(compiler, runner, ConsequenceLibraryBuilder().build(), MemoryMeter())


def a_row(played: RuleBasedGame) -> PositionRow:
    return PositionRow(played.start(), "X", 0.0)


def test_a_reading_gives_a_value_at_every_index_of_its_base(game: Game) -> None:
    played = game("tictactoe")

    values = caching().values(played, a_row(played), "cell", "1.0")

    assert len(values) == 9, "one per cell of the board"
    assert set(values) == {1.0}


def test_the_reading_is_read_over_the_index_it_is_written_about(game: Game) -> None:
    """A reading is Python over the index `i`, so what comes back is about each index in the base's order."""
    played = game("tictactoe")

    values = caching().values(played, a_row(played), "cell", "here.cell[i] is None")

    assert list(values) == [1.0] * 9, "an empty board, and a truth is a number here"


def test_the_same_reading_on_the_same_row_is_read_once(game: Game) -> None:
    """A generation's candidates are mostly one another's bodies with one more reading, so dozens ask for the
    same reading on the same rows."""
    played, cache = game("tictactoe"), caching()
    row = a_row(played)

    cache.values(played, row, "cell", "1.0")
    cache.values(played, row, "cell", "1.0")

    assert cache.memory_entries() == 1


def test_two_different_readings_are_kept_apart(game: Game) -> None:
    played, cache = game("tictactoe"), caching()
    row = a_row(played)

    assert cache.values(played, row, "cell", "1.0") != cache.values(played, row, "cell", "2.0")
    assert cache.memory_entries() == 2


def test_a_reading_that_cannot_be_read_here_gives_nothing_rather_than_raising(game: Game) -> None:
    """A generated candidate may name anything. One that names what is not there is not an error, it is a term
    the search prices out."""
    played = game("tictactoe")

    assert caching().values(played, a_row(played), "cell", "lamp") == ()
    assert caching().values(played, a_row(played), "lamp", "1.0") == ()


def test_a_reading_giving_something_that_is_not_a_finite_number_gives_nothing_there(game: Game) -> None:
    """None means it could not be read at that index, which a fit reads as blank rather than as zero."""
    played = game("tictactoe")

    values = caching().values(played, a_row(played), "cell", "'a word'")
    infinite = caching().values(played, a_row(played), "cell", "float('inf')")

    assert set(values) == {None}
    assert set(infinite) == {None}


def test_clearing_forgets_every_reading_kept(game: Game) -> None:
    played, cache = game("tictactoe"), caching()
    cache.values(played, a_row(played), "cell", "1.0")

    cache.clear()

    assert cache.memory_entries() == 0


def test_what_a_reading_gives_is_this_process_s_own(game: Game) -> None:
    """A copy sent elsewhere starts with nothing kept, and still reads."""
    played, cache = game("tictactoe"), caching()
    cache.values(played, a_row(played), "cell", "1.0")

    sent = pickle.loads(pickle.dumps(cache))

    assert cache.memory_entries() == 1
    assert sent.memory_entries() == 0
    assert set(sent.values(played, a_row(played), "cell", "1.0")) == {1.0}
