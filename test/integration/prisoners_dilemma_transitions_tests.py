from collections.abc import Callable
from dataclasses import replace

import pytest

from openmind.agent.constant.prisoners_dilemma_constant import STANDARD, VARIANTS
from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant
from openmind.agent.factory.prisoners_dilemma_factory import declare_prisoners_dilemma
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.model.action import Action
from openmind.world.model.state import State


type Game = Callable[[str], RuleBasedSystem]


def dilemma(knowledge: KnowledgeBase, variant: PrisonersDilemmaVariant) -> RuleBasedSystem:
    """The prisoner's dilemma variant declared under its own context, whether or not it is one of the known ones."""
    return create_rule_based_system(knowledge, declare_prisoners_dilemma(knowledge, variant))


def choose(choice: str) -> Action:
    return Action("choose", (("choice", choice),))


def play(knowledge: KnowledgeBase, variant: PrisonersDilemmaVariant, *choices: str) -> State:
    """The state after the choices, following the branch where the game goes on while there is one."""
    rbs = dilemma(knowledge, variant)
    state = rbs.start()
    for choice in choices:
        state = rbs.outcomes(state, choose(choice)).outcomes[0][0]
    return state


def legal_action_count(knowledge: KnowledgeBase, variant: PrisonersDilemmaVariant, state: State) -> int:
    return len(dilemma(knowledge, variant).actions(state))


def test_a_choice_stays_with_its_player_until_the_other_chooses(knowledge: KnowledgeBase, game: Game) -> None:
    state = play(knowledge, STANDARD, "defect")

    assert state.model("chosen") == Map.of({"A": "defect", "B": None})
    assert (state.model("played"), state.model("score")["A"], state.value("turn")) == (Grid.filled((1, 2), None), 0, "B")
    assert legal_action_count(knowledge, STANDARD, state) == 2


@pytest.mark.parametrize(
    ("first", "second", "points"),
    [("cooperate", "cooperate", (3, 3)), ("cooperate", "defect", (0, 5)), ("defect", "cooperate", (5, 0)), ("defect", "defect", (1, 1))],
)
def test_a_finished_round_plays_both_choices_adds_the_points_and_starts_the_next(knowledge: KnowledgeBase, game: Game, first: str, second: str, points: tuple[int, int]) -> None:
    state = play(knowledge, STANDARD, first, second)

    assert state.model("played") == Grid((2, 2), (first, second, None, None))
    assert state.model("score") == Map.of({"A": points[0], "B": points[1]})
    assert (state.model("chosen"), state.value("round"), state.value("turn")) == (Map.of({"A": None, "B": None}), 2, "A")
    assert state.model("payoff") == Map.of({"A": None, "B": None})


def test_after_the_last_round_each_payoff_is_the_player_s_own_total_and_the_game_is_over(knowledge: KnowledgeBase, game: Game) -> None:
    state = play(knowledge, STANDARD, *("defect", "cooperate") * 10)

    assert state.model("payoff") == Map.of({"A": 50, "B": 0})
    assert (state.model("played").shape, state.model("played")[10, 1], state.model("played")[10, 2]) == (
        (10, 2),
        "defect",
        "cooperate",
    )
    assert legal_action_count(knowledge, STANDARD, state) == 0


def test_without_a_known_last_round_a_finished_round_goes_on_or_ends_by_chance(knowledge: KnowledgeBase, game: Game) -> None:
    rbs = dilemma(knowledge, VARIANTS["uncertain"])
    chosen = play(knowledge, VARIANTS["uncertain"], "cooperate")

    (same, going_on_chance), (also_same, ending_chance) = rbs.outcomes(rbs.start(), choose("cooperate")).outcomes
    (goes_on, _), (ends, _) = rbs.outcomes(chosen, choose("cooperate")).outcomes

    assert (same, going_on_chance, ending_chance) == (also_same, pytest.approx(0.9), pytest.approx(0.1))
    assert (goes_on.value("round"), goes_on.model("payoff"), goes_on.model("played")[2, 1]) == (
        2,
        Map.of({"A": None, "B": None}),
        None,
    )
    assert (ends.value("round"), ends.model("payoff"), ends.model("played").shape) == (1, Map.of({"A": 3, "B": 3}), (1, 2))


def test_the_last_round_ends_the_game_on_either_branch_of_an_ending_chance(knowledge: KnowledgeBase, game: Game) -> None:
    variant = replace(STANDARD, rounds=2, ending_chance=0.5)
    rbs = dilemma(knowledge, variant)

    outcomes = rbs.outcomes(play(knowledge, variant, "cooperate", "defect", "defect"), choose("defect")).outcomes

    assert [outcome.model("payoff") for outcome, _ in outcomes] == [Map.of({"A": 1, "B": 6})] * 2
