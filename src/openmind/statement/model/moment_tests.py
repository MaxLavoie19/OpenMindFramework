from openmind.statement.model.literal import Literal
from openmind.statement.model.moment import (
    AFTER, AT, HAPPENS, MOMENTS, NOW, after, at, happens, moment, now, said,
)
from openmind.statement.model.term import Constant, Functor, Number


def test_saying_nothing_about_when_means_now():
    """Every reading there has ever been is of the position as it stands. Saying so is what lets moments arrive
    without a single existing clause having to be rewritten to mention one."""
    assert moment(None) == NOW
    assert Literal("holds", (Constant("a"),)).when is None


def test_a_moment_is_a_term_like_everything_else_that_is_said():
    assert isinstance(now(), Functor)
    assert at(3) == Functor(AT, (Number(3),))
    assert after(Constant("the move")) == Functor(AFTER, (Constant("the move"),))


def test_a_time_is_a_time_and_not_a_thing_that_happened():
    """Indexed by time because real time is coming: the situation calculus says outright that its actions have
    no duration, and next means the next turn, which a game without turns has not got."""
    assert moment(at(0)) == AT
    assert moment(at(1.5)) == AT, "a time need not be a whole number"


def test_after_is_kept_so_a_turn_based_game_need_not_invent_a_clock():
    """`after(A)` is `at(T + 1)` where a clock ticks once per action, and saying next should not require one."""
    assert moment(after(Constant("a move"))) == AFTER
    assert AFTER in MOMENTS


def test_something_can_happen_without_anybody_doing_it():
    """A clock running out, a pawn promoting, a pawn taken in passing. Having to say those as parts of somebody's
    move is what Consequence.order has been working around."""
    held = happens(Constant("the clock ran out"), at(7))

    assert held.name == HAPPENS
    assert held.arguments == (Constant("the clock ran out"), at(7))


def test_a_term_that_is_not_a_moment_is_not_taken_for_one():
    assert moment(Functor("place", (Constant("self"), Constant("row")))) == ""
    assert moment(Constant("now")) == "", "a name that reads like one is not one"


def test_the_same_thing_said_of_two_moments_is_two_statements():
    one = Literal("holds", (Constant("a"),))

    assert one != one.said_of(after(Constant("a move")))
    assert one.said_of(None) == one


def test_denying_something_keeps_when_it_was_said_of():
    """A king may not be left where it can be taken is one reading held now and denied later. If the denial lost
    the moment it would be a denial of something else."""
    later = Literal("holds", (Constant("a"),)).said_of(after(Constant("a move")))

    assert later.denied.when == later.when
    assert later.denied.negated != later.negated


def test_a_moment_reads_as_words():
    assert said(None) == "now"
    assert said(at(4)) == "at 4"
    assert said(after(Constant("a move"))).startswith("once ")
