import logging
import math
from collections.abc import Hashable
from dataclasses import replace

from openmind.agent.model.domain import Domain
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.hypothesis_result import HypothesisResult
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.model.theory_of_mind import TheoryOfMind
from openmind.mcts.service.tree_search import TreeSearch
from openmind.observation.service.state_observer import StateObserver
from openmind.timing.model.deadline import Deadline
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class SemiDeterminizedSearch:
    """Semi-determinized MCTS (Bitan and Kraus, 2017): the player to act asks its theory of mind for hypotheses about
    what it can't see, searches once as if each were true, with information set MCTS over that hypothesis's states, an
    even share of the iterations and an even share of the seconds still left, and weighs each action's mean payoffs by
    the hypotheses' probabilities. The action chosen has the highest expected payoff; a better theory of mind makes
    better choices."""

    def __init__(
        self,
        tree_search: TreeSearch,
        state_observer: StateObserver,
        state_reader: StateReader,
        action_text_mapper: ActionTextMapper,
    ) -> None:
        self._tree_search = tree_search
        self._state_observer = state_observer
        self._state_reader = state_reader
        self._action_text_mapper = action_text_mapper

    def search(
        self,
        domain: Domain,
        state: State,
        settings: SearchSettings,
        theory: TheoryOfMind,
        guidance: Guidance | None = None,
        valuation: LeafValuation | None = None,
    ) -> SearchResult:
        """A domain without an observation, no hypothesis, a negative probability, or probabilities that don't sum to 1
        raise ValueError."""
        observation = domain.observation
        if observation is None:
            raise ValueError("A semi-determinized search needs a domain with an observation")
        player = domain.players.names[self._state_reader.player_to_act(state, domain.players)]
        observed = self._state_observer.observe(observation, state, player)
        hypotheses = theory.hypotheses(domain, observed, player)
        if not hypotheses:
            raise ValueError(f"The theory of mind gave {player} no hypothesis")
        if negative := [probability for _, probability in hypotheses if probability < 0.0]:
            raise ValueError(f"A hypothesis for {player} has a negative probability, {negative[0]}")
        total = math.fsum(probability for _, probability in hypotheses)
        if not math.isclose(total, 1.0):
            raise ValueError(f"The hypotheses for {player} have probabilities summing to {total}, not 1")
        count = len(hypotheses)
        iterations = settings.iterations
        shares: list[int | None] = (
            [None] * count
            if iterations is None
            else [max(1, iterations // count + (index < iterations % count)) for index in range(count)]
        )
        logger.info(
            "%s weighs %d hypotheses: %s; %s",
            player,
            count,
            "; ".join(f"{_label_text(hypothesis.label)} at {probability}" for hypothesis, probability in hypotheses),
            _shares_text(shares, settings.seconds),
        )
        source = self._tree_search.time_source
        started = source.now()
        deadline = None if settings.seconds is None else Deadline(started + settings.seconds, source)
        searched: list[tuple[HypothesisResult, SearchResult]] = []
        for index, ((hypothesis, probability), share) in enumerate(zip(hypotheses, shares, strict=True)):
            shared = replace(settings, iterations=share, seed=None if settings.seed is None else settings.seed + index)
            logger.info("Searching as if %s", _label_text(hypothesis.label))
            if deadline is not None:
                left = deadline.remaining()
                if left > 0.0:
                    shared = replace(shared, seconds=left / (count - index))
                    logger.info("%s has %.3f seconds of the %.3f left for this hypothesis", player, shared.seconds, left)
                else:
                    shared = replace(shared, iterations=1, seconds=None)
                    logger.info("%s's time is up, so this hypothesis gets 1 iteration", player)
            result = self._tree_search.search(
                domain.problem,
                domain.transitions,
                domain.players,
                state,
                shared,
                guidance,
                valuation,
                observation,
                hypothesis.completions,
            )
            searched.append((HypothesisResult(hypothesis.label, probability, result.statistics), result))
        statistics = self._expected(searched)
        visited = [item for item in statistics if item.visits > 0]
        chosen = max(visited or statistics, key=lambda item: (item.mean_payoff, item.visits)).action
        logger.info(
            "Expected payoffs for %s: %s; chose %s",
            player,
            " ".join(f"{self._action_text_mapper.to_text(item.action)}={item.mean_payoff}" for item in statistics),
            self._action_text_mapper.to_text(chosen),
        )
        return SearchResult(
            player,
            statistics,
            chosen,
            tuple(sample for _, result in searched for sample in result.samples),
            tuple(hypothesis for hypothesis, _ in searched),
            iterations=sum(result.iterations for _, result in searched),
            seconds=source.now() - started,
        )

    def _expected(self, searched: list[tuple[HypothesisResult, SearchResult]]) -> tuple[ActionStatistics, ...]:
        """Every root action, in the order the searches first give them, with its visits summed and its mean payoffs
        weighed by the probabilities of the hypotheses whose search visited it."""
        by_hypothesis = [{item.action: item for item in hypothesis.statistics} for hypothesis, _ in searched]
        actions = dict.fromkeys(item.action for hypothesis, _ in searched for item in hypothesis.statistics)
        expected: list[ActionStatistics] = []
        for action in actions:
            visits, weights, payoffs = 0, [], []
            for (hypothesis, _), items in zip(searched, by_hypothesis, strict=True):
                item = items.get(action)
                if item is None or item.visits == 0:
                    continue
                visits += item.visits
                weights.append(hypothesis.probability)
                payoffs.append(hypothesis.probability * item.mean_payoff)
            weight = math.fsum(weights)
            expected.append(ActionStatistics(action, visits, math.fsum(payoffs) / weight if weight > 0 else 0.0))
        return tuple(expected)


def _label_text(label: Hashable) -> str:
    """A label of (name, value) pairs as `name='value', ...`; any other label as its repr."""
    if isinstance(label, tuple) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in label):
        return ", ".join(f"{name}={value!r}" for name, value in label)
    return repr(label)


def _shares_text(shares: list[int | None], seconds: float | None) -> str:
    """The hypotheses' shares as the log line says them: `20, 20 iterations`, `4 seconds shared as they go`, or both."""
    parts = [] if shares[0] is None else [f"{', '.join(map(str, shares))} iterations"]
    if seconds is not None:
        parts.append(f"{seconds:g} seconds shared as they go")
    return " and ".join(parts)
