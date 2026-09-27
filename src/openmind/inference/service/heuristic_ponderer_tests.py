import logging
from collections.abc import Callable

import pytest

from openmind.inference.factory.inference_factory import create_heuristic_ponderer
from openmind.inference.model.ponder_settings import PonderSettings
from openmind.inference.service.heuristic_ponderer import MOVES_SETTLED, REASONED, SETTLED
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.rule_based_game import RuleBasedGame

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedGame]


def settings(**held) -> PonderSettings:
    """Enough of a budget to reach every part of pondering, and no more of one than a test needs."""
    return PonderSettings(
        values=ValueSettings(
            prices=(0.0, 0.01), max_steps=100, tolerance=1e-4, seconds=3.0, memory_bytes=500_000_000
        ),
        positions=held.pop("positions", 30),
        held_out=held.pop("held_out", 10),
        plies=2,
        deduction_seconds=0.2,
        **held,
    )


def test_what_the_rules_imply_things_are_worth_seeds_the_search(game: Game, knowledge: KnowledgeBase, caplog):
    """The join, end to end: the rules are asked what a thing is worth and the answer arrives as a term.

    Nothing asked for marks. The reasoner was handed the game and gave back what taking each thing off the board
    costs its owner, the deriver turned that into a term counting them, and the search was handed the term with
    the number already on it."""
    caplog.set_level(logging.INFO, logger="openmind.inference")
    pondered = create_heuristic_ponderer(knowledge).ponder(knowledge, game("tictactoe"), settings())

    seeding = [one for one in pondered.tried if one.source == REASONED.format(context="tictactoe")]
    assert len(seeding) == 1
    assert any("seed the search with" in one or "seeds the search" in one for one in caplog.messages)
    assert any("terms seeded" in one for one in caplog.messages)


def test_a_mark_that_fills_a_square_is_seeded_at_a_worth_that_is_not_positive(game: Game, knowledge: KnowledgeBase):
    """A tic-tac-toe mark affords nothing — it takes a square away from whoever plays next, including its owner.

    So the reasoning that prices a chess knight by what it lets you do must price a mark at nothing or less, and
    it must do it without being told which game it is in. A positive worth here would mean the reasoning had
    found value in holding a thing that only ever costs its holder options, which is the failure this pins."""
    rules = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)

    worth = ponderer._worth.reason(rules, ponderer._gatherer.gather(rules, 20, 0))

    assert worth.holdings
    assert all(held <= 0 for _, _, held in worth.holdings), worth.holdings


def test_the_positions_kept_back_are_not_reasoned_over(game: Game, knowledge: KnowledgeBase, caplog):
    """The held-out rows choose which price is kept, so a seed reasoned out of them would make that choice partly
    a choice about rows it had already seen. What the rules imply is asked of the fitted-on positions only."""
    caplog.set_level(logging.INFO, logger="openmind.inference")
    ponderer = create_heuristic_ponderer(knowledge)

    pondered = ponderer.ponder(knowledge, game("tictactoe"), settings())

    valuing = next(one for one in pondered.tried if one.source == SETTLED.format(context="tictactoe"))
    seeding = next(one for one in pondered.tried if one.source == REASONED.format(context="tictactoe"))
    assert seeding.positions < valuing.positions


def test_a_budget_of_no_positions_reasons_over_none_and_still_reports_itself(game: Game, knowledge: KnowledgeBase):
    """A way of paying for a search that was given nothing to pay with says so, rather than vanishing.

    That is the same rule the ways of valuing already keep: a bootstrapper reads these to find out what to stop
    paying for, and a source that only appears when it succeeds cannot be read that way."""
    pondered = create_heuristic_ponderer(knowledge).ponder(
        knowledge, game("tictactoe"), settings(worth_positions=0)
    )

    seeding = next(one for one in pondered.tried if one.source == REASONED.format(context="tictactoe"))
    assert seeding.positions == 0
    assert not seeding.paid


def test_the_owning_structure_is_found_from_the_vocabulary_and_never_deduced(game: Game, knowledge: KnowledgeBase):
    """Whose a thing is comes out of the positions, not out of `SideDeducer`, which open question 33 has
    concluding that one player owns both light and dark squares.

    Tic-tac-toe keeps its marks and their owners in one grid — the mark *is* the player's name — so there is no
    second structure to find and the term reads `== me` directly. A game holding the two apart gets the
    two-condition term instead, and neither case is told which it is."""
    rules = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)
    positions = ponderer._gatherer.gather(rules, 20, 0)
    vocabulary = ponderer._expressions.vocabulary(rules, positions)

    assert ponderer._expressions.owning("cell", vocabulary) is None
    seeds = ponderer._deriver.seeds(ponderer._deriver.holdings(ponderer._worth.reason(rules, positions)), vocabulary)
    assert seeds
    assert all("== me" in one.template or "== other" in one.template for one, _ in seeds), seeds


def test_what_the_rules_settle_a_move_pays_gives_a_row_per_action(game: Game, knowledge: KnowledgeBase, caplog):
    """The other half of what the rules settle: a position is worth one thing to each player, an action is
    worth one thing to whoever takes it."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)
    positions = ponderer._gatherer.gather(played, 12, 1)
    tried = []

    with caplog.at_level(logging.INFO):
        rows = ponderer.moved(played, positions, settings(), tried)

    assert rows, "the rules of tic-tac-toe settle what some moves pay within two plies"
    assert all(row.player == played.acting_player(row.state) for row in rows), "valued for whoever acts"
    assert all(row.action in played.actions(row.state) for row in rows)
    assert any(one.source == MOVES_SETTLED.format(context=played.context) for one in tried), "it reports itself"


def test_a_move_nothing_proves_gets_no_row_rather_than_a_zero(game: Game, knowledge: KnowledgeBase):
    """Not knowing what a move pays is not the move paying nothing. A row saying zero would be a claim the
    evidence never made, and the fit would take it for one."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)
    positions = ponderer._gatherer.gather(played, 12, 1)

    rows = ponderer.moved(played, positions, settings(), [])

    for state in {row.state for row in rows}:
        rated = {row.action for row in rows if row.state is state}
        assert rated <= set(played.actions(state))


def test_settling_moves_reports_itself_even_where_it_settles_nothing(game: Game, knowledge: KnowledgeBase):
    """A way of valuing that taught nothing is a finding of its own, so it is in the labellings either way."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)
    tried = []

    rows = ponderer.moved(played, (), settings(), tried)

    assert rows == ()
    assert [one.source for one in tried] == [MOVES_SETTLED.format(context=played.context)]
    assert tried[0].paid is False


def test_valuing_keeps_the_positions_the_proofs_passed_through(game: Game, knowledge: KnowledgeBase, caplog):
    """Far more rows than positions asked about, for work already paid for: proving one position proves many,
    and all of it but the answer was being discarded."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)
    held = settings(positions=30, held_out=10)
    positions = ponderer._gatherer.gather(played, 40, 1)

    with caplog.at_level(logging.INFO):
        rows = ponderer._valued(played, played, positions, held, [])

    assert len({row.state for row in rows}) > len(positions), "more positions valued than were asked about"
    assert any("the proofs passed through" in one.message for one in caplog.records), "it says how many it kept"


def test_a_position_asked_about_keeps_the_value_it_was_asked_about(game: Game, knowledge: KnowledgeBase):
    """A note taken on the way to an answer never overrides the answer, so what the game paid a finished
    position stays what that position is worth."""
    from openmind.structure.model.grid import Grid
    from openmind.structure.model.map import Map
    from openmind.world.model.state import State

    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)
    finished = State.of(
        cell=Grid((3, 3), ("X", "X", "X", "O", "O", None, None, None, None)),
        turn="O",
        payoff=Map.of({"X": 1.0, "O": 0.0}),
    )
    positions = (*ponderer._gatherer.gather(played, 20, 1), finished)

    rows = ponderer._valued(played, played, positions, settings(), [])

    paid = {row.player: row.target for row in rows if row.state == finished}
    assert paid == {"X": 1.0, "O": 0.0}, "what the game paid, not what a walk noted"
