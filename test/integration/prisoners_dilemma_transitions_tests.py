from dataclasses import replace

import pytest

from openmind.agent.constant.prisoners_dilemma_constant import STANDARD, VARIANTS
from openmind.agent.factory.prisoners_dilemma_factory import create_prisoners_dilemma_domain
from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant
from openmind.csp.factory.csp_factory import create_solver
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value


def choose(choice: str) -> Action:
    return Action("choose", (("choice", choice),))


def play(variant: PrisonersDilemmaVariant, *choices: str) -> State:
    """The state after the choices, following the branch where the game goes on while there is one."""
    domain, predictor = create_prisoners_dilemma_domain(variant), create_predictor()
    state = domain.initial_state
    for choice in choices:
        state = predictor.predict(domain.transitions, state, choose(choice)).outcomes[0][0]
    return state


def values_of(state: State, *names: str) -> tuple[Value, ...]:
    variables = dict(state.variables)
    return tuple(variables[name] for name in names)


def legal_action_count(variant: PrisonersDilemmaVariant, state: State) -> int:
    return len(create_solver().solve(create_prisoners_dilemma_domain(variant).problem, state))


def test_a_choice_stays_with_its_player_until_the_other_chooses() -> None:
    state = play(STANDARD, "defect")

    assert values_of(state, "chosen(A)", "chosen(B)", "played(1,A)", "score(A)", "turn") == ("defect", None, None, 0, "B")
    assert legal_action_count(STANDARD, state) == 2


@pytest.mark.parametrize(
    ("first", "second", "points"),
    [("cooperate", "cooperate", (3, 3)), ("cooperate", "defect", (0, 5)), ("defect", "cooperate", (5, 0)), ("defect", "defect", (1, 1))],
)
def test_a_finished_round_plays_both_choices_adds_the_points_and_starts_the_next(first: str, second: str, points: tuple[int, int]) -> None:
    state = play(STANDARD, first, second)

    assert values_of(state, "played(1,A)", "played(1,B)", "score(A)", "score(B)") == (first, second, *points)
    assert values_of(state, "chosen(A)", "chosen(B)", "round", "played(2,A)", "played(2,B)", "turn") == (None, None, 2, None, None, "A")
    assert values_of(state, "payoff(A)", "payoff(B)") == (None, None)


def test_after_the_last_round_each_payoff_is_the_player_s_own_total_and_the_game_is_over() -> None:
    state = play(STANDARD, *("defect", "cooperate") * 10)

    assert values_of(state, "payoff(A)", "payoff(B)", "played(10,A)", "played(10,B)") == (50, 0, "defect", "cooperate")
    assert "played(11,A)" not in dict(state.variables)
    assert legal_action_count(STANDARD, state) == 0


def test_without_a_known_last_round_a_finished_round_goes_on_or_ends_by_chance() -> None:
    domain, predictor = create_prisoners_dilemma_domain(VARIANTS["uncertain"]), create_predictor()
    chosen = play(VARIANTS["uncertain"], "cooperate")

    (same, going_on_chance), (also_same, ending_chance) = predictor.predict(domain.transitions, domain.initial_state, choose("cooperate")).outcomes
    (goes_on, _), (ends, _) = predictor.predict(domain.transitions, chosen, choose("cooperate")).outcomes

    assert (same, going_on_chance, ending_chance) == (also_same, pytest.approx(0.9), pytest.approx(0.1))
    assert values_of(goes_on, "round", "payoff(A)", "payoff(B)", "played(2,A)") == (2, None, None, None)
    assert values_of(ends, "round", "payoff(A)", "payoff(B)") == (1, 3, 3)
    assert "played(2,A)" not in dict(ends.variables)


def test_the_last_round_ends_the_game_on_either_branch_of_an_ending_chance() -> None:
    variant = replace(STANDARD, rounds=2, ending_chance=0.5)
    domain = create_prisoners_dilemma_domain(variant)

    outcomes = create_predictor().predict(domain.transitions, play(variant, "cooperate", "defect", "defect"), choose("defect")).outcomes

    assert [values_of(outcome, "payoff(A)", "payoff(B)") for outcome, _ in outcomes] == [(1, 6), (1, 6)]
