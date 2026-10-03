import logging
from collections.abc import Callable

import pytest

from openmind.inference.factory.inference_factory import create_heuristic_ponderer
from openmind.inference.model.ponder_settings import PonderSettings
from openmind.inference.service.heuristic_ponderer import MOVES_SETTLED, SETTLED
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.model.value_settings import ValueSettings
from openmind.knowledge.constant.rule_kind_constant import MOVE
from openmind.knowledge.constant.task_constant import MOVE_VALUE
from openmind.rbs.factory.rbs_factory import create_rule_based_system, find_rule_based_system
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


def test_the_rows_held_back_are_gathered_positions_and_not_nodes_from_a_tree(
    game: Game, knowledge: KnowledgeBase
):
    """The price is chosen on the rows held back, and the nodes a proof passed through outnumber the gathered
    positions many times over. Held back at the end, the choice would be made on endgames — where a proof is
    cheap — rather than on the positions the heuristic is for."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)
    positions = ponderer._gatherer.gather(played, 40, 1)

    rows = ponderer._valued(played, played, positions, settings(), [])
    _, held_out = ponderer._split(rows, 10)

    assert held_out, "something is held back"
    asked = set(positions)
    assert all(row.state in asked for row in held_out), "every row held back is a position that was gathered"


def test_steadiness_is_measured_only_where_it_is_asked_for(game: Game, knowledge: KnowledgeBase, caplog):
    """Off by default, because measuring it made held-out loss worse on every seed tried. A caller that wants
    it asks for walks; one that does not is not quietly given it."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)

    with caplog.at_level(logging.INFO):
        ponderer.ponder(knowledge, played, settings())

    assert not [one for one in caplog.records if "kept and" in one.message], "nothing measured, nothing dropped"


def test_walking_orders_the_terms_and_drops_none_of_them(game: Game, knowledge: KnowledgeBase, caplog):
    """A leaf that carries nothing by itself carries plenty as a part, and dropping it takes everything the
    search would have grown from it. Measured, that cost between two and thirteen times the held-out loss
    while removing a twentieth of the candidates, so nothing is dropped unless a budget asks."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)

    with caplog.at_level(logging.INFO):
        pondered = ponderer.ponder(knowledge, played, settings(walks=3))

    assert pondered is not None
    said = [one.message for one in caplog.records if "kept and" in one.message]
    assert said, "it reports what it did"
    assert " 0 dropped" in said[0], said[0]


def test_the_steadiest_budget_lets_fewer_terms_through(game: Game, knowledge: KnowledgeBase, caplog):
    """A budget the caller set, not a level this picked: asked for two, the search is told about the rest."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)

    with caplog.at_level(logging.INFO):
        ponderer.ponder(knowledge, played, settings(walks=3, steadiest=2))

    kept = [one.message for one in caplog.records if "kept and" in one.message]
    assert kept, "it reports the split"
    assert " 2 kept and " in kept[0], kept[0]


def test_a_move_heuristic_is_fitted_where_the_position_heuristic_is(game: Game, knowledge: KnowledgeBase, caplog):
    """**Both halves of what a search needs, learned in one pass.** A search walks by what a move is rated and
    stops where a position is valued, and until now only the second was ever fitted — the first was derived a
    ply ahead from it, which costs a valuing per move and is the expense a move heuristic exists to remove.

    Fitted here rather than anywhere else because the positions are already gathered and already deduced over;
    fitting elsewhere would walk them twice and let the two halves drift apart over different games."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)
    positions = ponderer._gatherer.gather(played, 8, 1)  # noqa: SLF001
    # What a search settled on: every legal action, the first of them favoured. The shape is what a strategy
    # carries, which is what travels from a worker that searched.
    settled = {}
    for state in positions:
        actions = played.actions(state)
        if actions:
            share = 1.0 / (len(actions) + 1)
            settled[state] = tuple(
                (action, 2 * share if at == 0 else share) for at, action in enumerate(actions)
            )

    with caplog.at_level(logging.INFO):
        ponderer.ponder(knowledge, played, settings(positions=8, held_out=2), positions, settled=settled)

    found = create_rule_based_system(knowledge, played.context, MOVE_VALUE)
    assert found.rules, "a move value ruleset of its own, not rules written into the position value's"
    assert all(rule.kind == MOVE for rule, _ in found.rules)
    # **A move rule that reads no action rates every move in a position alike**, which is no policy at all.
    # Written as a reading off the view rather than as the bare name it is bound under, every such term
    # raised, its column came back as nothing, and the fit kept only terms that read the board -- a move
    # heuristic that passed every other assertion here and could not tell one move from another.
    named = {"action"} | {name for held in settled.values() for action, _ in held for name, _ in action.parameters}
    assert any(
        any(one in rule.rule.source for one in named) for rule, _ in found.rules
    ), "at least one fitted rule reads the action and not only the board"


def test_nothing_is_fitted_for_moves_where_no_search_settled_anything(game: Game, knowledge: KnowledgeBase):
    """A run that never searched has nothing that says which move was worth exploring. What a game paid says
    who won later and what the rules prove a move pays is a reward, and neither is a policy — so the honest
    answer is no move heuristic rather than one fitted on a stand-in."""
    played = game("tictactoe")
    ponderer = create_heuristic_ponderer(knowledge)

    ponderer.ponder(knowledge, played, settings(positions=6, held_out=2))

    assert find_rule_based_system(knowledge, played.context, MOVE_VALUE) is None
