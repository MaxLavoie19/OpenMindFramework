from collections.abc import Callable
from dataclasses import replace

import pytest

from openmind.agent.constant.prisoners_dilemma_constant import STANDARD, VARIANTS
from openmind.agent.factory.prisoners_dilemma_factory import (
    create_prisoners_dilemma_definitions,
    create_prisoners_dilemma_initial_state,
    create_prisoners_dilemma_players,
    declare_prisoners_dilemma,
)
from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant
from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT, EFFECTS, VALUES
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.testing.plugin.game_fixtures import simulation_rules
from openmind.world.model.players import Players
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]


def test_initial_state_is_round_1_with_nothing_chosen_or_played_and_a_to_choose() -> None:
    assert create_prisoners_dilemma_initial_state() == State.of(
        chosen=Map.of({"A": None, "B": None}),
        played=Grid.filled((1, 2), None),
        score=Map.of({"A": 0, "B": 0}),
        payoff=Map.of({"A": None, "B": None}),
        round=1,
        turn="A",
    )


def test_a_simultaneous_variant_has_no_turn_and_both_players_choose_at_once(game: Game) -> None:
    rbs = game("prisonersdilemma/simultaneous")

    assert not rbs.start().has("turn")
    assert rbs.acting(rbs.start()) == (0, 1)


def test_a_player_chooses_to_cooperate_or_defect_once_a_round_while_no_payoff_is_set(
    knowledge: KnowledgeBase,
) -> None:
    context = declare_prisoners_dilemma(knowledge)

    (choice,) = simulation_rules(knowledge, context, (VALUES,))
    constraints = simulation_rules(knowledge, context, (CONSTRAINT,))

    assert (choice.action, choice.parameter) == ("choose", "choice")
    assert choice.rule == PythonRule("('cooperate', 'defect')")
    assert [rule.rule for rule in constraints] == [
        PythonRule("turn == player"),
        PythonRule("payoff['A'] is None"),
        PythonRule("payoff['B'] is None"),
        PythonRule("chosen[player] is None"),
    ]


@pytest.mark.parametrize(("name", "rounds"), [("standard", 10), ("uncertain", None)])
def test_definitions_give_axelrod_s_points_and_the_rounds(name: str, rounds: int | None) -> None:
    reading = RuleCompiler().compile_value(
        PythonRule("(POINTS['cooperate', 'cooperate'], POINTS['cooperate', 'defect'], POINTS['defect', 'defect'], ROUNDS, other('A'))"),
        (),
        create_prisoners_dilemma_definitions(VARIANTS[name]),
    )

    assert RuleRunner(StateNamespaceMapper()).value(reading, State(())) == ((3, 3), (0, 5), (1, 1), rounds, "B")


@pytest.mark.parametrize(
    ("variant", "probabilities"),
    [(STANDARD, (1.0,)), (VARIANTS["uncertain"], (0.9, 0.1)), (replace(STANDARD, ending_chance=1.0), (1.0,))],
)
def test_a_round_s_outcomes_carry_the_ending_chance(
    knowledge: KnowledgeBase, variant: PrisonersDilemmaVariant, probabilities: tuple[float, ...]
) -> None:
    context = declare_prisoners_dilemma(knowledge, variant)

    outcomes = simulation_rules(knowledge, context, (EFFECTS,))

    assert {rule.action for rule in outcomes} == {"choose"}
    assert tuple(rule.probability for rule in outcomes) == pytest.approx(probabilities)


def test_players_are_a_and_b_with_the_payoff_map() -> None:
    assert create_prisoners_dilemma_players() == Players(("A", "B"), "payoff")


def test_a_variant_is_declared_under_its_own_name_and_the_standard_game_keeps_its_name(
    knowledge: KnowledgeBase,
) -> None:
    assert declare_prisoners_dilemma(knowledge, VARIANTS["uncertain"]) == "prisonersdilemma/uncertain"
    assert declare_prisoners_dilemma(knowledge) == "prisonersdilemma"


def test_the_declared_game_is_played_round_by_round(game: Game) -> None:
    rbs = game("prisonersdilemma")
    state = rbs.start()

    for _ in range(4):
        state = rbs.outcomes(state, rbs.actions(state)[0]).outcomes[0][0]

    assert state.value("round") == 3
    assert state.model("score")["A"] == 6
    assert state.model("played") == Grid((3, 2), ("cooperate", "cooperate", "cooperate", "cooperate", None, None))


@pytest.mark.parametrize(
    "variant",
    [
        replace(STANDARD, rounds=0),
        replace(STANDARD, ending_chance=-0.1),
        replace(STANDARD, ending_chance=1.5),
        replace(STANDARD, rounds=None, ending_chance=0.0),
    ],
)
def test_a_variant_without_a_way_to_end_raises(knowledge: KnowledgeBase, variant: PrisonersDilemmaVariant) -> None:
    with pytest.raises(ValueError, match="A prisoner's dilemma variant needs at least 1 round"):
        declare_prisoners_dilemma(knowledge, variant)
