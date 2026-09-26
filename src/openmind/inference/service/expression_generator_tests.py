from openmind.inference.model.expression import Expression
from openmind.inference.model.vocabulary import Vocabulary
from openmind.inference.service.expression_generator import ExpressionGenerator

PLAYERS = ("me", "you")


def a_vocabulary(**held) -> Vocabulary:
    """A game of two grids over the same squares: what stands there, and whose it is.

    The shape chess has, and the one the whole seeding argument turns on — a value alone counts both sides'
    knights, and only the second condition counts mine.
    """
    places = frozenset({(1, 1), (1, 2), (2, 1), (2, 2)})
    fields = {
        "players": PLAYERS,
        "values_by_variable": {("turn", ()): ("me", "you")},
        "values_by_base": {"piece": ("knight", "rook"), "colour": ("me", "you"), "clock": (1, 2, 3)},
        "indices_by_base": {"piece": places, "colour": places, "clock": places},
        "offsets_by_arity": {2: ((0, 0), (0, 1), (1, 0))},
        "grids": frozenset({"piece", "colour"}),
    }
    return Vocabulary(**{**fields, **held})


def templates(expressions) -> list[str]:
    return [one.template for one in expressions]


def test_counting_a_value_gives_how_many_there_are_and_how_many_are_mine():
    """The second is the one worth anything and the search reaches it late: counting knights counts both sides'
    at once, which barely moves from position to position, so the search passes over it — while the term whose
    weight *is* what a knight is worth needs the ownership condition."""
    found = templates(ExpressionGenerator().counting("piece", "knight", a_vocabulary(), "colour"))

    assert len(found) == 2
    assert any("== 'knight'" in one and "colour" not in one for one in found)
    assert any("== 'knight'" in one and "colour" in one and "me" in one for one in found)


def test_counting_without_an_owner_gives_only_the_plain_count():
    found = templates(ExpressionGenerator().counting("piece", "knight", a_vocabulary()))

    assert len(found) == 1
    assert "colour" not in found[0]


def test_an_owner_over_different_places_is_not_used():
    """Two structures are only about the same things where they are indexed alike; otherwise pairing them says
    a knight is mine because something unrelated of mine sits at the same number."""
    vocabulary = a_vocabulary(
        indices_by_base={"piece": frozenset({(1, 1), (1, 2)}), "colour": frozenset({(9, 9)}), "clock": frozenset({(1, 1)})}
    )

    found = ExpressionGenerator().counting("piece", "knight", vocabulary, "colour")

    assert len(found) == 1


def test_counting_something_the_base_never_holds_gives_nothing():
    """A thing a game does not have is not a thing to count, and a term counting it would read nought forever."""
    assert ExpressionGenerator().counting("piece", "dragon", a_vocabulary(), "colour") == ()
    assert ExpressionGenerator().counting("nowhere", "knight", a_vocabulary(), "colour") == ()


def test_which_base_says_whose_a_thing_is_falls_out_of_the_vocabulary():
    """Worked out and never declared, so a game keeping ownership somewhere else, or nowhere, is not told it
    does — it is the base over the same places whose values are the players' names."""
    assert ExpressionGenerator().owning("piece", a_vocabulary()) == "colour"


def test_a_game_that_says_whose_nothing_is_has_no_owning_base():
    vocabulary = a_vocabulary(values_by_base={"piece": ("knight",), "colour": ("light", "dark"), "clock": (1,)})

    assert ExpressionGenerator().owning("piece", vocabulary) is None


def test_a_game_with_no_players_has_no_owning_base():
    assert ExpressionGenerator().owning("piece", a_vocabulary(players=())) is None


def test_a_numeric_base_gets_aggregates_and_no_pattern_of_its_values():
    """A number is not a thing to count the instances of: what is worth reading about a clock is its total, its
    least and its most, not how many squares hold a three."""
    found = templates(ExpressionGenerator().leaves(a_vocabulary()))

    assert not any("clock" in one and "== 3" in one for one in found)
    assert any("clock" in one and one.startswith("sum(") for one in found)


def test_the_leaves_offer_what_each_player_can_do():
    """Mobility is the one reading that is about the game rather than about the board, and every game has it."""
    found = templates(ExpressionGenerator().leaves(a_vocabulary()))

    assert any("mobility(me)" in one for one in found)
    assert any("mobility(other)" in one for one in found)


def test_a_threshold_never_cuts_where_every_value_falls_on_one_side():
    """A term that is true of everything read is a term that says nothing, and it costs a clause to say it."""
    found = ExpressionGenerator().thresholds(Expression("x", 1, 0), [1.0, 2.0, 3.0])

    cuts = {(relation, cut) for _, relation, cut in found}
    assert (">=", 1.0) not in cuts  # everything is at least the lowest
    assert ("<=", 3.0) not in cuts  # everything is at most the highest
    assert (">=", 2.0) in cuts and ("<=", 1.0) in cuts


def test_a_threshold_takes_each_value_once_however_often_it_was_read():
    found = ExpressionGenerator().thresholds(Expression("x", 1, 0), [1.0, 1.0, 1.0, 2.0])

    assert len({cut for _, _, cut in found}) == 2


def test_dividing_never_divides_by_less_than_one():
    """A term reading nought is a term a game will meet, and a heuristic that raises there is a heuristic the
    search quietly drops instead of one somebody fixes."""
    found = dict((operator, one.template) for one, operator in ExpressionGenerator().combinations(
        Expression("a", 1, 0), Expression("b", 1, 0)
    ))

    assert found["/"] == "(a) / max(1, b)"


def test_combining_two_expressions_costs_both_their_clauses():
    """What a term costs is what it reads, so a combination that was priced as one clause would be cheaper than
    its own parts and the sweep would keep it for that reason alone."""
    found = ExpressionGenerator().combinations(Expression("a", 2, 0), Expression("b", 3, 1))

    assert all(one.clauses == 5 for one, _ in found)
    assert all(one.plies == 1 for one, _ in found)


def test_looking_ahead_reads_the_position_after_an_action_and_costs_a_ply():
    """A look-ahead is the only kind of term that makes the search play a move to read it, which is why it
    carries its plies — a planner is told how far ahead its heuristic already looks."""
    found = ExpressionGenerator().look_aheads(Expression("here_x", 1, 0))

    assert all(one.plies == 1 for one in found)
    assert any(".best(me" in one.template for one in found)
    assert any(".worst(other" in one.template for one in found)


def test_a_look_ahead_of_a_look_ahead_takes_a_variable_of_its_own():
    """Two look-aheads sharing a variable would read the same position twice and call it two moves."""
    once = ExpressionGenerator().look_aheads(Expression("x", 1, 0))[0]

    twice = ExpressionGenerator().look_aheads(once)[0]

    assert twice.plies == 2
    assert "v1" in twice.template and "v2" in twice.template


def test_an_expression_is_read_as_a_rule_over_the_position_in_hand():
    """The templates are written over a view so a look-ahead can rebind it; a rule reads the position itself."""
    assert ExpressionGenerator().source(Expression("{view}.mobility(me)", 1, 0)).source == "here.mobility(me)"
