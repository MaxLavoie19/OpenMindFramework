import numpy as np
import pytest

from openmind.inference.model.pattern import Pattern
from openmind.inference.model.pattern_condition import PatternCondition
from openmind.inference.model.vocabulary import Vocabulary
from openmind.inference.service.term_assembler import TermAssembler
from openmind.rbs.model.position_row import PositionRow
from openmind.structure.model.grid import Grid
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")

PLACES = frozenset({(1, 1), (1, 2), (2, 1), (2, 2)})


def a_vocabulary() -> Vocabulary:
    """Two grids over the same places: what stands there, and whose it is. The shape chess has."""
    return Vocabulary(
        players=("me", "you"),
        values_by_variable={},
        values_by_base={"thing": ("stone", "stick"), "whose": ("me", "you")},
        indices_by_base={"thing": PLACES, "whose": PLACES},
        offsets_by_arity={2: ((0, 1), (1, 0))},
        grids=frozenset({"thing", "whose"}),
    )


def a_row(things, whose, player="me") -> PositionRow:
    """A two-by-two board, written out as two grids of four."""
    return PositionRow(
        State.of(
            thing=Grid.of([list(things[:2]), list(things[2:])]),
            whose=Grid.of([list(whose[:2]), list(whose[2:])]),
        ),
        player,
        0.0,
    )


def boards(how_many: int = 12):
    """Boards holding between none and four stones of mine, and what each is worth: how many there are."""
    rows, worth = [], []
    for at in range(how_many):
        count = at % 5
        things = ["stone"] * count + [None] * (4 - count)
        rows.append(a_row(things, ["me"] * count + [None] * (4 - count)))
        worth.append(float(count))
    return rows, worth


def test_a_term_is_assembled_from_what_the_rows_are_missing():
    """The whole of it: shown boards and a number per board, it says what the boards with the larger number
    have that the others do not."""
    rows, worth = boards()

    found = TermAssembler().assembled(rows, worth, a_vocabulary())

    assert found, "something accounts for a count that varies with what stands on the board"
    assert found[0].pattern is not None


def test_a_term_nothing_can_improve_is_still_the_term_it_gives_back():
    """**The bug this cost a session to find.** A beam built only from what could be extended drops a term the
    moment nothing improves it — which is exactly when it is finished. Measured against a real target: a
    one-condition term reading it perfectly was thrown away for a three-condition term correlating at 0.169,
    and it read as the assembly ignoring the vocabulary it had been given rather than as a beam losing its
    own best answer.

    Here one condition is the whole answer, so the first round settles it and no later round can do better."""
    rows, worth = boards()

    found = TermAssembler().assembled(rows, worth, a_vocabulary())

    assert found[0].clauses == 1, "the one condition that says it, and nothing hung on the end of it"


def test_an_idiom_is_one_condition_where_the_cells_cannot_say_it_at_all():
    """**Why there are idioms at all.** A term is only short in the words it is written in, and every search
    in this literature is capped between four and six conditions — so a term that is a dozen conditions over
    cells is not slow to reach, it is out of reach. Named and offered back, it is one condition.

    The idiom here reads something the grids do not hold: whether a place is an even one. That is the shape of
    the real case, where an idiom reads what the game's own move rules allow and no grid records — which
    square can be taken, which is defended. Cells can say what stands where and never that."""
    rows, worth = [], []
    for at in range(12):
        # Stones on the two even places, on the two odd ones, on all four, or on none.
        held = (((1, 1), (2, 2)), ((1, 2), (2, 1)), ((1, 1), (2, 2), (1, 2), (2, 1)), ())[at % 4]
        things = ["stone" if (r, c) in held else None for r in (1, 2) for c in (1, 2)]
        rows.append(a_row(things, ["me" if one else None for one in things]))
        worth.append(float(sum(1 for r, c in held if (r + c) % 2 == 0)))
    idiom = ("a stone of mine on an even place", "evenly placed")

    def says(row, source, where):
        return (where[0] + where[1]) % 2 == 0 and row.state.model("thing").at(where) == "stone"

    found = TermAssembler().assembled(rows, worth, a_vocabulary(), idioms=(idiom,), says=says)

    assert found, "the idiom says exactly what is being counted"
    assert [one.name for one in found[0].pattern.conditions] == ["a stone of mine on an even place"]


def test_an_assembled_term_can_be_said_in_words_so_a_thing_with_no_name_has_one():
    """A term has to be called something before the next round can speak of it, and the only thing known
    about one the moment it is assembled is what it reads. Conditions about one place are said together,
    because a kind and an owner are two conditions meaning one thing."""
    pattern = Pattern(
        "thing",
        (
            PatternCondition("thing", (0, 0), "==", "'stone'"),
            PatternCondition("whose", (0, 0), "==", "me"),
        ),
        ((
            PatternCondition("thing", (1, 0), "==", "'stone'"),
            PatternCondition("whose", (1, 0), "==", "other"),
        ),),
    )

    assert TermAssembler().worded(pattern) == "stone of mine here, and no stone of theirs at (1, 0)"


def test_an_idiom_keeps_the_name_it_was_given_rather_than_being_described_again():
    """It was named when it earned its keep; describing its insides again would lose what it is called."""
    pattern = Pattern("thing", (PatternCondition("", (), "", None, None, "x", "a stone I can take"),))

    assert TermAssembler().worded(pattern) == "a stone I can take"


def test_rows_and_what_they_are_missing_have_to_be_the_same_positions():
    with pytest.raises(ValueError, match="not the same positions"):
        TermAssembler().assembled(boards()[0], [1.0], a_vocabulary())


def test_nothing_is_assembled_where_there_is_nothing_to_account_for():
    """A target that never varies has nothing a term could explain, and a term offered for it would be an
    ordering invented out of an absence."""
    rows, _ = boards()

    assert TermAssembler().assembled(rows, [1.0] * len(rows), a_vocabulary()) == ()


def test_a_name_says_what_a_thing_is_before_it_says_whose_it_is():
    """**The words in the right sentence.** Conditions arrive in whatever order the assembly found them, and a
    name read off in that order says "of mine pawn" — the right words, badly ordered. Here the owner is given
    first, as a real assembly often does, and the name still reads as a person would say it."""
    pattern = Pattern(
        "thing",
        (
            PatternCondition("whose", (0, 0), "==", "me"),
            PatternCondition("thing", (0, 0), "==", "'stone'"),
        ),
    )

    assert TermAssembler().worded(pattern) == "stone of mine here"


def test_a_place_known_only_by_whose_it_is_still_reads_as_a_sentence():
    """Nothing says what stands there, only that it is somebody's, and the name should say that rather than
    begin mid-phrase."""
    pattern = Pattern("thing", (PatternCondition("whose", (1, 0), "==", "other"),))

    assert TermAssembler().worded(pattern) == "something of theirs at (1, 0)"
