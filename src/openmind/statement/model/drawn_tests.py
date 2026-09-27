from openmind.statement.model.drawn import (
    ALWAYS, DRAWINGS, PLACE, STEPPED, always, asked, column, drawing, more, other, part, place, row,
    said, standing, stepped,
)
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Variable
from openmind.inference.service.unifier import Unifier


def test_a_drawing_is_a_term():
    """The whole of stage one. A drawing and a reading's argument were different kinds of thing, so the same
    square had two spellings and nothing knew they were one square."""
    assert isinstance(place("self", "row"), Functor)
    assert place("self", "row") == Functor(PLACE, (Constant("self"), Constant("row")))


def test_which_way_of_drawing_a_term_is_can_be_asked_of_it():
    assert drawing(stepped("self", "row", "x")) == STEPPED
    assert drawing(always("a thing")) == ALWAYS
    assert drawing(Functor("something else", ())) == "", "not one of them"
    assert drawing(Constant("self")) == "", "not even a functor"


def test_every_way_of_drawing_is_in_the_one_list():
    """So anything reading them by name reads one list rather than keeping its own."""
    for built in (place("p", "q"), stepped("p", "q", "r"), row("p"), column("p"),
                  standing("m", "p"), asked("p"), always(1), other(), more("m", 2)):
        assert drawing(built) in DRAWINGS


def test_a_drawing_says_its_parts_as_the_plain_values_they_stand_for():
    assert part(place("self", "row"), 0) == "self"
    assert part(place("self", "row"), 1) == "row"
    assert part(more("clock", 3), 1) == 3, "a number comes back a number"
    assert part(other(), 0) is None, "asked for a part it has not got"


def test_a_drawing_unifies_like_any_other_term():
    """What it is for. A rule about the place a move starts can be said once, with a variable, because a
    drawing is in the language variables live in."""
    made = Unifier().unify(
        Literal("at", (place("self", "row"),)),
        Literal("at", (Functor(PLACE, (Variable("Which"), Constant("row"))),)),
    )

    assert made is not None
    assert made.applied(Variable("Which")) == Constant("self")


def test_a_drawing_reads_as_words_for_anything_showing_one():
    assert said(place("self", "row")) == "the row of self"
    assert said(stepped("self", "row", "x")) == "the row of self, stepped by x"
    assert said(standing("grid", "self")) == "what grid holds at self"
    assert said(other()) == "the player not acting"
    assert said(more("clock", 2)) == "clock and 2 more"
