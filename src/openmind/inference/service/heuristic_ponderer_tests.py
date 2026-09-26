import logging
from collections.abc import Callable

import pytest

from openmind.inference.factory.inference_factory import create_heuristic_ponderer
from openmind.inference.model.ponder_settings import PonderSettings
from openmind.inference.service.heuristic_ponderer import REASONED, SETTLED
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
