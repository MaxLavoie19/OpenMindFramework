from dataclasses import replace

import pytest

from openmind.agent.constant.prisoners_dilemma_constant import STANDARD, VARIANTS
from openmind.agent.factory.prisoners_dilemma_factory import (
    create_prisoners_dilemma_definitions,
    create_prisoners_dilemma_domain,
    create_prisoners_dilemma_initial_state,
    create_prisoners_dilemma_observation,
    create_prisoners_dilemma_players,
    create_prisoners_dilemma_problem,
    create_prisoners_dilemma_transitions,
)
from openmind.agent.model.domain import Domain
from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State


def test_initial_state_is_round_1_with_nothing_chosen_or_played_and_a_to_choose() -> None:
    variables = dict(create_prisoners_dilemma_initial_state().variables)

    assert variables == {
        "chosen(A)": None,
        "chosen(B)": None,
        "played(1,A)": None,
        "played(1,B)": None,
        "score(A)": 0,
        "score(B)": 0,
        "payoff(A)": None,
        "payoff(B)": None,
        "round": 1,
        "turn": "A",
    }


def test_problem_chooses_to_cooperate_or_defect_once_a_round_while_no_payoff_is_set() -> None:
    assert create_prisoners_dilemma_problem() == Problem(
        (
            ActionDefinition(
                "choose",
                (Variable("choice", DiscreteDomain(("cooperate", "defect"))),),
                (PythonRule("payoff['A'] is None"), PythonRule("payoff['B'] is None"), PythonRule("chosen[turn] is None")),
            ),
        ),
    )


@pytest.mark.parametrize(("name", "rounds"), [("standard", 10), ("uncertain", None)])
def test_definitions_give_axelrod_s_points_and_the_rounds(name: str, rounds: int | None) -> None:
    reading = RuleCompiler().compile_value(
        PythonRule("(POINTS['cooperate', 'cooperate'], POINTS['cooperate', 'defect'], POINTS['defect', 'defect'], ROUNDS, other('A'))"),
        (),
        create_prisoners_dilemma_definitions(VARIANTS[name]),
    )

    assert RuleRunner(StateNamespaceMapper(VariableNameMapper())).value(reading, State(())) == ((3, 3), (0, 5), (1, 1), rounds, "B")


@pytest.mark.parametrize(
    ("variant", "probabilities"),
    [(STANDARD, (1.0,)), (VARIANTS["uncertain"], (0.9, 0.1)), (replace(STANDARD, ending_chance=1.0), (1.0,))],
)
def test_transitions_branch_on_the_ending_chance(variant: PrisonersDilemmaVariant, probabilities: tuple[float, ...]) -> None:
    (transition,) = create_prisoners_dilemma_transitions(variant).transitions

    assert transition.action == "choose"
    assert tuple(branch.probability for branch in transition.branches) == pytest.approx(probabilities)


def test_players_are_a_and_b_with_turn_and_payoff_variables() -> None:
    assert create_prisoners_dilemma_players() == Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)"))


def test_domain_holds_the_recipes_and_a_variant_is_named_after_the_game() -> None:
    uncertain = VARIANTS["uncertain"]

    assert create_prisoners_dilemma_domain(uncertain) == Domain(
        "prisonersdilemma/uncertain",
        create_prisoners_dilemma_initial_state(),
        create_prisoners_dilemma_problem(),
        create_prisoners_dilemma_transitions(uncertain),
        create_prisoners_dilemma_players(),
        create_prisoners_dilemma_observation(uncertain),
    )
    assert create_prisoners_dilemma_domain().name == "prisonersdilemma"


def test_a_player_can_t_see_the_other_s_choice_which_could_be_either_once_made() -> None:
    observer, observation = create_state_observer(), create_prisoners_dilemma_observation()
    variables = dict(create_prisoners_dilemma_initial_state().variables)
    after_a_defected = State(tuple(sorted({**variables, "chosen(A)": "defect", "turn": "B"}.items())))

    seen_by_b = observer.observe(observation, after_a_defected, "B")
    seen_by_a = observer.observe(observation, after_a_defected, "A")

    assert (dict(seen_by_b.variables)["chosen(A)"], dict(seen_by_a.variables)["chosen(A)"]) == ("<hidden>", "defect")
    assert [
        (dict(state.variables)["chosen(A)"], probability) for state, probability in observer.completions(observation, seen_by_b, "B")
    ] == [("cooperate", 0.5), ("defect", 0.5)]
    assert [
        (dict(state.variables)["chosen(B)"], probability) for state, probability in observer.completions(observation, seen_by_a, "A")
    ] == [(None, 1.0)]


@pytest.mark.parametrize(
    "variant",
    [
        replace(STANDARD, rounds=0),
        replace(STANDARD, ending_chance=-0.1),
        replace(STANDARD, ending_chance=1.5),
        replace(STANDARD, rounds=None, ending_chance=0.0),
    ],
)
def test_a_variant_without_a_way_to_end_raises(variant: PrisonersDilemmaVariant) -> None:
    with pytest.raises(ValueError, match="A prisoner's dilemma variant needs at least 1 round"):
        create_prisoners_dilemma_domain(variant)
