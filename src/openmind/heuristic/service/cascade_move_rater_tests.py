from openmind.heuristic.model.node import Node
from openmind.heuristic.service.cascade_move_rater import CascadeMoveRater
from openmind.world.model.action import Action

ACTIONS = (Action("place", (("col", 1),)), Action("place", (("col", 2),)))


class _Says:
    """A rater that says what it was told to, and None where it was told nothing."""

    def __init__(self, *rated: float | None) -> None:
        self._rated = rated

    def rate(self, model, node, actions, player):
        return self._rated[: len(actions)]


def test_the_first_rater_with_anything_to_say_answers() -> None:
    """A rater that has to answer everywhere is one model pretending to be a library."""
    cascade = CascadeMoveRater(
        ((None, _Says(None, None)), (None, _Says(0.7, 0.2))), ("the table", "the rules")
    )

    assert cascade.rate(None, Node(None), ACTIONS, "X") == (0.7, 0.2)


def test_a_rater_that_speaks_answers_the_whole_decision_and_not_part_of_it() -> None:
    """Forced, not chosen. Two raters' numbers are on unrelated scales — one fitted on payoffs, another on
    how often somebody chose a move — so taking one rater's score for this action and another's for that one
    compares numbers that were never comparable."""
    cascade = CascadeMoveRater(((None, _Says(0.9, None)), (None, _Says(0.1, 0.8))))

    rated = cascade.rate(None, Node(None), ACTIONS, "X")

    assert rated == (0.9, None), "the second rater is not consulted for the action the first passed over"


def test_where_every_rater_declines_so_does_the_cascade() -> None:
    """Nobody knowing anything is a thing to say, and saying nothing is how it is said."""
    cascade = CascadeMoveRater(((None, _Says(None, None)), (None, _Says(None, None))))

    assert cascade.rate(None, Node(None), ACTIONS, "X") == (None, None)


def test_a_cascade_says_which_of_its_raters_answered() -> None:
    """A caller asking who decided is asking a different question from what to play, and the answer is what
    makes the decision readable rather than merely correct."""
    cascade = CascadeMoveRater(
        ((None, _Says(None, None)), (None, _Says(0.7, 0.2))), ("the table", "the rules")
    )

    assert cascade.answered(Node(None), ACTIONS, "X") == "the rules"
    assert cascade.layers == ("the table", "the rules")


def test_nobody_answering_is_named_as_nobody() -> None:
    cascade = CascadeMoveRater(((None, _Says(None, None)),), ("the table",))

    assert cascade.answered(Node(None), ACTIONS, "X") == ""


def test_a_cascade_of_cascades_is_a_cascade() -> None:
    """So cascades compose and nothing downstream can tell one rater from a stack of them."""
    inner = CascadeMoveRater(((None, _Says(None, None)), (None, _Says(0.5, 0.4))), ("a", "b"))
    outer = CascadeMoveRater(((None, _Says(None, None)), (None, inner)), ("c", "the inner one"))

    assert outer.rate(None, Node(None), ACTIONS, "X") == (0.5, 0.4)
    assert outer.answered(Node(None), ACTIONS, "X") == "the inner one"
