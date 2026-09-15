import pytest

from openmind.agent.factory.prisoners_dilemma_factory import create_prisoners_dilemma_domain
from openmind.agent.model.domain import Domain
from openmind.agent.service.completion_theory import CompletionTheory
from openmind.mcts.service.semi_determinized_search_tests import coin_domain
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action


def new_theory(label: PythonRule | None = None) -> CompletionTheory:
    return CompletionTheory(create_state_observer(), create_rule_caller(), label)


def test_the_second_prisoner_weighs_the_first_s_hidden_choice_as_the_domain_does() -> None:
    domain = create_prisoners_dilemma_domain()
    ((state, _),) = create_predictor().predict(
        domain.transitions, domain.initial_state, Action("choose", (("choice", "cooperate"),))
    ).outcomes
    observed = create_state_observer().observe(domain.observation, state, "B")  # type: ignore[arg-type]

    hypotheses = new_theory().hypotheses(domain, observed, "B")

    assert sorted((hypothesis.label, probability) for hypothesis, probability in hypotheses) == [
        ((("chosen(A)", "cooperate"),), 0.5),
        ((("chosen(A)", "defect"),), 0.5),
    ]
    assert all(len(hypothesis.completions) == 1 and hypothesis.completions[0][1] == 1.0 for hypothesis, _ in hypotheses)


def test_a_label_rule_says_what_the_hypotheses_are_about_and_renormalizes_their_states() -> None:
    domain = coin_domain("tails", 0.9)
    observed = create_state_observer().observe(domain.observation, domain.initial_state, "me")  # type: ignore[arg-type]

    hypotheses = new_theory(PythonRule("coin == 'heads'")).hypotheses(domain, observed, "me")

    assert [(hypothesis.label, probability) for hypothesis, probability in hypotheses] == [(True, 0.9), (False, pytest.approx(0.1))]
    assert [hypothesis.completions[0][1] for hypothesis, _ in hypotheses] == [1.0, 1.0]


def test_a_domain_without_an_observation_raises() -> None:
    domain = coin_domain("tails", 0.5)
    plain = Domain(domain.name, domain.initial_state, domain.problem, domain.transitions, domain.players)

    with pytest.raises(ValueError, match="needs a domain with an observation"):
        new_theory().hypotheses(plain, domain.initial_state, "me")
