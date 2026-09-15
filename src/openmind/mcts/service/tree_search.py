import logging
import math
import random
from collections.abc import Iterator

from openmind.csp.model.problem import Problem
from openmind.csp.service.solver import Solver
from openmind.mcts.model.action_sample import ActionSample
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.chance_node import ChanceNode
from openmind.mcts.model.decision_node import DecisionNode
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.observation.model.observation import Observation
from openmind.observation.service.state_observer import StateObserver
from openmind.predictor.model.transition_model import TransitionModel
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)

#: How a search with an observation sees: the observation, the searching player's name, and the states that could be
#: be true at the root with their probabilities.
type View = tuple[Observation, str, tuple[tuple[State, float], ...]]


class TreeSearch:
    """Monte-Carlo Tree Search: UCT selection, chance nodes for outcomes, rollouts, optional guidance by a rater, and
    optional valuation of the positions rollouts reach. With an observation, the searching player sees only what it
    shows them: single-observer information set MCTS."""

    def __init__(
        self,
        solver: Solver,
        predictor: Predictor,
        state_reader: StateReader,
        action_text_mapper: ActionTextMapper,
        state_observer: StateObserver | None = None,
    ) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._action_text_mapper = action_text_mapper
        self._state_observer = create_state_observer() if state_observer is None else state_observer

    def search(
        self,
        problem: Problem,
        transitions: TransitionModel,
        players: Players,
        state: State,
        settings: SearchSettings,
        guidance: Guidance | None = None,
        valuation: LeafValuation | None = None,
        observation: Observation | None = None,
    ) -> SearchResult:
        if settings.rollout_limit is not None:
            if settings.rollout_limit < 0:
                raise ValueError(f"The rollout limit can't be negative, not {settings.rollout_limit}")
            if settings.unfinished_payoff is None:
                raise ValueError("A rollout limit needs an unfinished payoff")
        rng = random.Random(settings.seed)
        view: View | None = None
        if observation is not None:
            searching = players.names[self._state_reader.player_to_act(state, players)]
            state = self._state_observer.observe(observation, state, searching)
            view = (observation, searching, self._state_observer.completions(observation, state, searching))
        root = self._decision_node(problem, players, state, guidance, rng)
        if root.player is None:
            raise ValueError("No legal action to search from")
        player = players.names[root.player]
        logger.info("Searching %d iterations for %s", settings.iterations, player)
        if view is not None:
            logger.info("%s sees %d states that could be true", player, len(view[2]))
        for iteration in range(1, settings.iterations + 1):
            self._iterate(iteration, root, problem, transitions, players, settings, guidance, valuation, rng, view)
        statistics = tuple(self._statistics(root, root.player, action) for action in root.actions)
        chosen = max(statistics, key=lambda item: item.visits).action
        for item in statistics:
            logger.info(
                "%s: %d visits, mean payoff %s for %s",
                self._action_text_mapper.to_text(item.action),
                item.visits,
                item.mean_payoff,
                player,
            )
        logger.info("Most visited: %s", self._action_text_mapper.to_text(chosen))
        return SearchResult(player, statistics, chosen, tuple(self._samples(root, view is not None)))

    def _iterate(
        self,
        iteration: int,
        root: DecisionNode,
        problem: Problem,
        transitions: TransitionModel,
        players: Players,
        settings: SearchSettings,
        guidance: Guidance | None,
        valuation: LeafValuation | None,
        rng: random.Random,
        view: View | None,
    ) -> None:
        """One iteration. With a view, it draws a state that could be true at the root and plays on it: legal actions and
        outcomes come from that state, and a node is what the searching player sees of the states reaching it."""
        node = root
        state = root.state if view is None else self._draw(view[2], rng)
        decisions = [root]
        chances: list[ChanceNode] = []
        while actions := node.actions if view is None else self._solver.solve(problem, state):
            action = self._untried(node, actions, view is not None, rng)
            if action is not None:
                outcomes = self._predictor.predict(transitions, state, action).outcomes
                chance = ChanceNode(action, outcomes, {}, 0, [0.0] * len(players.names))
                node.children[action] = chance
            else:
                chance = node.children[self._select(node, actions, settings.exploration, guidance)]
                outcomes = (
                    chance.outcomes
                    if view is None
                    else self._predictor.predict(transitions, state, chance.action).outcomes
                )
            chances.append(chance)
            state = self._draw(outcomes, rng)
            node = self._outcome(chance, state, problem, players, guidance, rng, view)
            decisions.append(node)
            if node.visits == 0:
                actions = node.actions
                break

        if not actions:
            payoffs, rollout_length, ending = self._state_reader.payoffs(state, players), 0, ""
        else:
            payoffs, rollout_length, ending = self._rollout(
                problem, transitions, players, state, settings, guidance, valuation, rng
            )

        for decision in decisions:
            decision.visits += 1
        for chance in chances:
            chance.visits += 1
            for index, payoff in enumerate(payoffs):
                chance.payoff_sums[index] += payoff

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Iteration %d: %s, rollout of %d actions%s, payoffs %s",
                iteration,
                " > ".join(self._action_text_mapper.to_text(chance.action) for chance in chances),
                rollout_length,
                ending,
                " ".join(f"{name}={payoff}" for name, payoff in zip(players.names, payoffs)),
            )

    def _untried(
        self, node: DecisionNode, actions: tuple[Action, ...], observed: bool, rng: random.Random
    ) -> Action | None:
        """The next action to try, or None when every legal action has been tried. With an observation, legal actions
        can differ between the states drawn: the untried ones still come in the node's order, and an action the node
        didn't have when created comes in random order after them."""
        if not observed:
            return node.untried.pop() if node.untried else None
        fresh = [action for action in actions if action not in node.children]
        if not fresh:
            return None
        for index in range(len(node.untried) - 1, -1, -1):
            if node.untried[index] in fresh:
                return node.untried.pop(index)
        return rng.choice(fresh)

    def _select(
        self, node: DecisionNode, actions: tuple[Action, ...], exploration: float, guidance: Guidance | None
    ) -> Action:
        log_visits = math.log(node.visits)
        prior_weight = guidance.prior_weight if guidance is not None and node.ratings else 0.0

        def score(action: Action, rating: float) -> float:
            chance = node.children[action]
            mean = chance.payoff_sums[node.player] / chance.visits  # type: ignore[index]
            explore = exploration * math.sqrt(log_visits / chance.visits)
            prior = prior_weight * rating / (chance.visits + 1) if prior_weight else 0.0
            return mean + explore + prior

        if actions is node.actions:
            return node.actions[
                max(
                    range(len(node.actions)),
                    key=lambda index: score(node.actions[index], node.ratings[index] if prior_weight else 0.0),
                )
            ]
        rating_of = dict(zip(node.actions, node.ratings)) if node.ratings else {}
        return max(
            (action for action in actions if action in node.children),
            key=lambda action: score(action, rating_of.get(action, 0.0)),
        )

    def _outcome(
        self,
        chance: ChanceNode,
        state: State,
        problem: Problem,
        players: Players,
        guidance: Guidance | None,
        rng: random.Random,
        view: View | None,
    ) -> DecisionNode:
        seen = state if view is None else self._state_observer.observe(view[0], state, view[1])
        child = chance.children.get(seen)
        if child is None:
            child = self._decision_node(problem, players, state, guidance, rng, seen)
            chance.children[seen] = child
        return child

    def _decision_node(
        self,
        problem: Problem,
        players: Players,
        state: State,
        guidance: Guidance | None,
        rng: random.Random,
        seen: State | None = None,
    ) -> DecisionNode:
        """A node for the state, holding what the searching player sees of it, rated on that."""
        actions = self._solver.solve(problem, state)
        shown = state if seen is None else seen
        ratings = self._ratings(guidance, shown, actions) if actions else ()
        untried = list(actions)
        rng.shuffle(untried)
        if ratings:
            rating_of = dict(zip(actions, ratings, strict=True))
            untried.sort(key=rating_of.__getitem__)
        player = self._state_reader.player_to_act(state, players) if actions else None
        return DecisionNode(shown, actions, untried, {}, 0, player, ratings)

    def _ratings(self, guidance: Guidance | None, state: State, actions: tuple[Action, ...]) -> tuple[float, ...]:
        """The rater's ratings, with the mean of the known ones for actions it can't rate; empty when unguided."""
        if guidance is None:
            return ()
        rated = guidance.rater.rate(state, actions)
        known = [rating for rating in rated if rating is not None]
        if not known:
            return ()
        fill = math.fsum(known) / len(known)
        return tuple(fill if rating is None else rating for rating in rated)

    def _rollout(
        self,
        problem: Problem,
        transitions: TransitionModel,
        players: Players,
        state: State,
        settings: SearchSettings,
        guidance: Guidance | None,
        valuation: LeafValuation | None,
        rng: random.Random,
    ) -> tuple[tuple[float, ...], int, str]:
        """Each player's payoff, the rollout's length, and how it ended, for the log. After the valuation's rollout
        actions, a position still in play gets the valuer's payoffs, unless the valuer knows nothing about it; at the
        rollout limit, every player gets the unfinished payoff."""
        length = 0
        while actions := self._solver.solve(problem, state):
            if valuation is not None and length == valuation.rollout_actions:
                values = valuation.valuer.value(state)
                if values is not None:
                    return values, length, ", then valued"
            if settings.rollout_limit is not None and length >= settings.rollout_limit:
                unfinished = float(settings.unfinished_payoff)  # type: ignore[arg-type]
                return (unfinished,) * len(players.names), length, ", then stopped at the rollout limit"
            ratings = self._ratings(guidance, state, actions) if guidance is not None and guidance.guided_rollouts else ()
            if guidance is not None and ratings:
                best = max(ratings)
                weights = [math.exp((rating - best) / guidance.rollout_temperature) for rating in ratings]
                (action,) = rng.choices(actions, weights=weights)
            else:
                action = rng.choice(actions)
            outcomes = self._predictor.predict(transitions, state, action).outcomes
            state = self._draw(outcomes, rng)
            length += 1
        return self._state_reader.payoffs(state, players), length, ""

    def _draw(self, outcomes: tuple[tuple[State, float], ...], rng: random.Random) -> State:
        (state,) = rng.choices(
            [outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes]
        )
        return state

    def _statistics(self, root: DecisionNode, player: int, action: Action) -> ActionStatistics:
        chance = root.children.get(action)
        if chance is None or chance.visits == 0:
            return ActionStatistics(action, 0, 0.0)
        return ActionStatistics(action, chance.visits, chance.payoff_sums[player] / chance.visits)

    def _samples(self, root: DecisionNode, observed: bool) -> Iterator[ActionSample]:
        """A sample for every expanded action; with an observation, also those of actions a node didn't have when
        created."""
        pending = [root]
        while pending:
            node = pending.pop()
            if node.player is None:
                continue
            expanded = node.actions
            if observed:
                expanded = (*node.actions, *(action for action in node.children if action not in node.actions))
            for action in expanded:
                chance = node.children.get(action)
                if chance is None or chance.visits == 0:
                    continue
                mean = chance.payoff_sums[node.player] / chance.visits
                yield ActionSample(node.state, node.player, action, chance.visits, mean)
                pending.extend(chance.children.values())
