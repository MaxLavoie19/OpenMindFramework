import logging
import math

import pytest

from openmind.agent.model.domain import Domain
from openmind.agent.service.completion_theory import CompletionTheory
from openmind.csp.factory.csp_factory import create_solver
from openmind.mcts.model.hypothesis import Hypothesis
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.semi_determinized_search import SemiDeterminizedSearch
from openmind.mcts.service.tree_search import TreeSearch
from openmind.mcts.service.tree_search_tests import guessing_game, hidden_coin
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

HEADS = Action("guess", (("side", "heads"),))
TAILS = Action("guess", (("side", "tails"),))
SETTINGS = SearchSettings(40, math.sqrt(2), 1)


class Believes:
    """A theory of mind believing the coin shows heads with the given chance, whatever the domain says."""

    def __init__(self, heads: float) -> None:
        self._heads = heads

    def hypotheses(self, domain: Domain, observed: State, player: str) -> tuple[tuple[Hypothesis, float], ...]:
        def showing(side: str) -> Hypothesis:
            state = State(tuple((name, side if name == "coin" else value) for name, value in observed.variables))
            return Hypothesis((("coin", side),), ((state, 1.0),))

        return (showing("heads"), self._heads), (showing("tails"), 1.0 - self._heads)

    def strategy(self, domain: Domain, state: State, player: str, other: str) -> None:
        return None


def coin_domain(secret: str, heads: float) -> Domain:
    """The guessing game as a domain in which me can't see the coin, heads with the given chance."""
    problem, transitions, state = guessing_game(secret)
    return Domain("coin", state, problem, transitions, Players(("me",), "turn", ("payoff",)), hidden_coin(heads))


def new_search() -> SemiDeterminizedSearch:
    return SemiDeterminizedSearch(
        TreeSearch(create_solver(), create_predictor(), StateReader(), ActionTextMapper()),
        create_state_observer(),
        StateReader(),
        ActionTextMapper(),
    )


def completion_theory() -> CompletionTheory:
    return CompletionTheory(create_state_observer(), RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())))


def test_a_theory_believing_heads_guesses_heads_where_the_domain_gives_even_chances() -> None:
    domain = coin_domain("tails", 0.5)

    result = new_search().search(domain, domain.initial_state, SETTINGS, Believes(0.8))

    expected = {item.action: item.mean_payoff for item in result.statistics}
    assert result.chosen == HEADS
    assert (expected[HEADS], expected[TAILS]) == (pytest.approx(0.8), pytest.approx(0.2))
    assert [(hypothesis.label, hypothesis.probability) for hypothesis in result.hypotheses] == [
        ((("coin", "heads"),), 0.8),
        ((("coin", "tails"),), pytest.approx(0.2)),
    ]
    assert sum(item.visits for item in result.statistics) == SETTINGS.iterations


def test_believing_the_domain_s_completions_weighs_each_hidden_value_as_the_domain_does(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    domain = coin_domain("tails", 0.5)

    result = new_search().search(domain, domain.initial_state, SETTINGS, completion_theory())

    assert {item.action: item.mean_payoff for item in result.statistics} == {HEADS: 0.5, TAILS: 0.5}
    assert "me weighs 2 hypotheses: coin='heads' at 0.5; coin='tails' at 0.5; 20, 20 iterations" in caplog.messages
    assert "Searching as if coin='tails'" in caplog.messages
    assert any(message.startswith("Expected payoffs for me: guess(side='heads')=0.5") for message in caplog.messages)


def test_the_search_is_the_same_whatever_the_hidden_value() -> None:
    heads, tails = coin_domain("heads", 0.5), coin_domain("tails", 0.5)

    assert new_search().search(heads, heads.initial_state, SETTINGS, Believes(0.7)) == new_search().search(
        tails, tails.initial_state, SETTINGS, Believes(0.7)
    )


class Overconfident(Believes):
    """A theory of mind giving heads 0.8 and tails 0.4."""

    def hypotheses(self, domain: Domain, observed: State, player: str) -> tuple[tuple[Hypothesis, float], ...]:
        (heads, _), (tails, _) = super().hypotheses(domain, observed, player)
        return (heads, 0.8), (tails, 0.4)


def test_hypotheses_whose_probabilities_don_t_sum_to_1_raise() -> None:
    domain = coin_domain("tails", 0.5)

    with pytest.raises(ValueError, match="summing to 1.2"):
        new_search().search(domain, domain.initial_state, SETTINGS, Overconfident(0.8))


def test_a_hypothesis_with_a_negative_probability_raises() -> None:
    domain = coin_domain("tails", 0.5)

    with pytest.raises(ValueError, match="negative probability"):
        new_search().search(domain, domain.initial_state, SETTINGS, Believes(1.2))


def test_a_domain_without_an_observation_raises() -> None:
    problem, transitions, state = guessing_game("tails")
    domain = Domain("coin", state, problem, transitions, Players(("me",), "turn", ("payoff",)))

    with pytest.raises(ValueError, match="needs a domain with an observation"):
        new_search().search(domain, state, SETTINGS, Believes(0.5))
