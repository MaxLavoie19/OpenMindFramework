from collections.abc import Callable
from dataclasses import replace

import pytest

from openmind.agent.constant.prisoners_dilemma_constant import STANDARD, VARIANTS
from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant
from openmind.agent.factory.prisoners_dilemma_factory import declare_prisoners_dilemma
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value


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


def values_of(state: State, *names: str) -> tuple[Value, ...]:
    variables = dict(state.variables)
    return tuple(variables[name] for name in names)


def legal_action_count(knowledge: KnowledgeBase, variant: PrisonersDilemmaVariant, state: State) -> int:
    return len(dilemma(knowledge, variant).actions(state))


def test_a_choice_stays_with_its_player_until_the_other_chooses(knowledge: KnowledgeBase, game: Game) -> None:
    state = play(knowledge, STANDARD, "defect")

    assert values_of(state, "chosen(A)", "chosen(B)", "played(1,A)", "score(A)", "turn") == ("defect", None, None, 0, "B")
    assert legal_action_count(knowledge, STANDARD, state) == 2


@pytest.mark.parametrize(
    ("first", "second", "points"),
    [("cooperate", "cooperate", (3, 3)), ("cooperate", "defect", (0, 5)), ("defect", "cooperate", (5, 0)), ("defect", "defect", (1, 1))],
)
def test_a_finished_round_plays_both_choices_adds_the_points_and_starts_the_next(knowledge: KnowledgeBase, game: Game, first: str, second: str, points: tuple[int, int]) -> None:
    state = play(knowledge, STANDARD, first, second)

    assert values_of(state, "played(1,A)", "played(1,B)", "score(A)", "score(B)") == (first, second, *points)
    assert values_of(state, "chosen(A)", "chosen(B)", "round", "played(2,A)", "played(2,B)", "turn") == (None, None, 2, None, None, "A")
    assert values_of(state, "payoff(A)", "payoff(B)") == (None, None)


def test_after_the_last_round_each_payoff_is_the_player_s_own_total_and_the_game_is_over(knowledge: KnowledgeBase, game: Game) -> None:
    state = play(knowledge, STANDARD, *("defect", "cooperate") * 10)

    assert values_of(state, "payoff(A)", "payoff(B)", "played(10,A)", "played(10,B)") == (50, 0, "defect", "cooperate")
    assert "played(11,A)" not in dict(state.variables)
    assert legal_action_count(knowledge, STANDARD, state) == 0


def test_without_a_known_last_round_a_finished_round_goes_on_or_ends_by_chance(knowledge: KnowledgeBase, game: Game) -> None:
    rbs = dilemma(knowledge, VARIANTS["uncertain"])
    chosen = play(knowledge, VARIANTS["uncertain"], "cooperate")

    (same, going_on_chance), (also_same, ending_chance) = rbs.outcomes(rbs.start(), choose("cooperate")).outcomes
    (goes_on, _), (ends, _) = rbs.outcomes(chosen, choose("cooperate")).outcomes

    assert (same, going_on_chance, ending_chance) == (also_same, pytest.approx(0.9), pytest.approx(0.1))
    assert values_of(goes_on, "round", "payoff(A)", "payoff(B)", "played(2,A)") == (2, None, None, None)
    assert values_of(ends, "round", "payoff(A)", "payoff(B)") == (1, 3, 3)
    assert "played(2,A)" not in dict(ends.variables)


def test_the_last_round_ends_the_game_on_either_branch_of_an_ending_chance(knowledge: KnowledgeBase, game: Game) -> None:
    variant = replace(STANDARD, rounds=2, ending_chance=0.5)
    rbs = dilemma(knowledge, variant)

    outcomes = rbs.outcomes(play(knowledge, variant, "cooperate", "defect", "defect"), choose("defect")).outcomes

    assert [values_of(outcome, "payoff(A)", "payoff(B)") for outcome, _ in outcomes] == [(1, 6), (1, 6)]
