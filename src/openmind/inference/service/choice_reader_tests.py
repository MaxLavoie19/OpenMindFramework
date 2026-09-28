from openmind.inference.service.choice_reader import ChoiceReader
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.model.action import Action
from openmind.world.model.state import State


class _Reads:
    """Readings a test dictates, so what is pinned here is the reading into columns and not the vocabulary."""

    def __init__(self, said: dict[tuple[int, ...], tuple[str, ...]]) -> None:
        self._said = said

    def read(self, state, action):
        return tuple(_Said(one) for one in self._said.get(tuple(value for _, value in action.parameters), ()))


class _Said:
    def __init__(self, readable: str) -> None:
        self.readable = readable


def _placing(column: int) -> Action:
    return Action("place", (("col", column),))


def _board() -> State:
    return State.of(cell=Grid((3, 3), (None,) * 9), turn="X", payoff=Map.of({"X": None, "O": None}))


def test_a_decision_becomes_a_row_per_candidate_and_a_column_per_reading() -> None:
    """The readings are the terms and nothing has to be cut anywhere: a reading of a candidate holds or it
    does not, which is already the shape a rule wants."""
    reader = ChoiceReader(_Reads({(1,): ("takes something",), (2,): ("moves forward",)}))

    choices, columns = reader.read([(_board(), (_placing(1), _placing(2)), 0)])

    assert columns == ("moves forward", "takes something")
    assert choices[0].candidates.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert choices[0].taken == 0


def test_a_reading_that_never_tells_two_candidates_apart_is_dropped() -> None:
    """Arithmetic rather than a judgement. The same number added to every candidate of a decision cancels in
    the choosing, so such a column contributes nothing whatever weight it is given."""
    reader = ChoiceReader(
        _Reads({(1,): ("is a move", "takes something"), (2,): ("is a move",)})
    )

    _, columns = reader.read([(_board(), (_placing(1), _placing(2)), 0)])

    assert columns == ("takes something",), "'is a move' holds for both, so it cannot bear on the choice"


def test_a_reading_that_varies_somewhere_is_kept_even_where_it_does_not(
) -> None:
    """It says nothing about the decision where it is constant and something about the one where it is not,
    and dropping it would lose the second to spare the first."""
    reader = ChoiceReader(
        _Reads({(1,): ("takes something",), (2,): ("takes something",), (3,): ()})
    )

    _, columns = reader.read(
        [
            (_board(), (_placing(1), _placing(2)), 0),  # both take: it says nothing here
            (_board(), (_placing(1), _placing(3)), 0),  # one takes: it says something here
        ]
    )

    assert columns == ("takes something",)


def test_a_decision_with_one_thing_on_offer_is_not_a_decision() -> None:
    reader = ChoiceReader(_Reads({(1,): ("takes something",)}))

    assert reader.read([(_board(), (_placing(1),), 0)]) == ((), ())


def test_nothing_telling_any_candidate_from_another_reads_as_nothing() -> None:
    """Said rather than returned as empty columns nobody can fit on."""
    reader = ChoiceReader(_Reads({(1,): ("is a move",), (2,): ("is a move",)}))

    assert reader.read([(_board(), (_placing(1), _placing(2)), 0)]) == ((), ())


def test_the_columns_come_back_in_a_settled_order() -> None:
    """So two runs over the same games put the same term in the same column, which is what lets a fitted
    weight be compared with another run's."""
    reader = ChoiceReader(_Reads({(1,): ("b", "a"), (2,): ("c",)}))

    _, columns = reader.read([(_board(), (_placing(1), _placing(2)), 0)])

    assert columns == ("a", "b", "c")
