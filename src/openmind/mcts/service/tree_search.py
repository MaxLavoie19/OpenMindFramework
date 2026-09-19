import logging
import math
import random
from collections.abc import Callable, Iterator, Mapping

from openmind.debug.factory.debugger_factory import process_debugger
from openmind.mcts.constant.mcts_constant import PUCT, SELECTIONS
from openmind.mcts.model.action_sample import ActionSample
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.chance_node import ChanceNode
from openmind.mcts.model.decision_node import DecisionNode
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.model.simultaneous_node import SimultaneousNode
from openmind.timing.model.deadline import Deadline
from openmind.timing.model.time_source import TimeSource
from openmind.timing.service.wall_time_source import WallTimeSource
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)

#: The states that could be true at the root, with their probabilities: what a hypothesis says the position may be.
#: be true at the root with their probabilities.
type Possible = tuple[tuple[State, float], ...]
#: What players acting at once picked at a node: each player's action index with the probability it was sampled at.
type Picked = tuple[tuple[int, float], ...]


class TreeSearch:
    """Monte-Carlo Tree Search: UCT selection, chance nodes for outcomes, rollouts, optional guidance by a rater, and
    optional valuation of the positions rollouts reach. Given the states a position could be, such as one hypothesis
    of a theory of mind, each iteration draws one of them and plays on it. Where players act at once, each samples its
    action by regret matching, and the searching player's action is sampled from its average strategy."""

    def __init__(
        self,
        state_reader: StateReader,
        action_text_mapper: ActionTextMapper,
        time_source: TimeSource | None = None,
    ) -> None:
        self._state_reader = state_reader
        self._action_text_mapper = action_text_mapper
        self._time_source = WallTimeSource() if time_source is None else time_source

    @property
    def time_source(self) -> TimeSource:
        """Where the search's seconds are read from."""
        return self._time_source

    def search(
        self,
        rbs: RuleBasedGame,
        state: State,
        settings: SearchSettings,
        guidance: Guidance | None = None,
        valuation: LeafValuation | None = None,
        possible: Possible | None = None,
        player: str | None = None,
        predicted: Mapping[str, tuple[tuple[Action, float], ...]] | None = None,
    ) -> SearchResult:
        """Searches the state, under a `search` reasoning frame; see `_search`."""
        with process_debugger().frame("search", context=rbs.context, state=state):
            return self._search(rbs, state, settings, guidance, valuation, possible, player, predicted)

    def _search(
        self,
        rbs: RuleBasedGame,
        state: State,
        settings: SearchSettings,
        guidance: Guidance | None = None,
        valuation: LeafValuation | None = None,
        possible: Possible | None = None,
        player: str | None = None,
        predicted: Mapping[str, tuple[tuple[Action, float], ...]] | None = None,
    ) -> SearchResult:
        """`possible` gives the states the position could be, such as one hypothesis of a theory of mind, each
        iteration drawing one of them and playing on it. In a state where players act at once, `player` names the
        searching player, and `predicted` gives other players to act the strategies they play at the root instead of
        regret matching, as (action, probability) pairs."""
        if settings.iterations is None and settings.seconds is None:
            raise ValueError("A search needs iterations, seconds or both")
        if settings.iterations is not None and settings.iterations < 1:
            raise ValueError(f"A search needs at least 1 iteration, not {settings.iterations}")
        if settings.seconds is not None and settings.seconds <= 0.0:
            raise ValueError(f"A search needs more than 0 seconds, not {settings.seconds}")
        if settings.selection not in SELECTIONS:
            raise ValueError(f"A search selects by {' or '.join(SELECTIONS)}, not {settings.selection!r}")
        if settings.puct_exploration < 0.0:
            raise ValueError(f"PUCT's exploration can't be negative, not {settings.puct_exploration}")
        if settings.rollout_limit is not None:
            if settings.rollout_limit < 0:
                raise ValueError(f"The rollout limit can't be negative, not {settings.rollout_limit}")
            if settings.unfinished_payoff is None:
                raise ValueError("A rollout limit needs an unfinished payoff")
        rng = random.Random(settings.seed)
        if len(rbs.acting(state)) > 1:
            if not 0.0 < settings.regret_exploration <= 1.0:
                raise ValueError(f"The regret exploration needs to be above 0 and at most 1, not {settings.regret_exploration}")
            return self._search_at_once(rbs, state, settings, valuation, player, predicted or {}, rng)
        root = self._decision_node(rbs, state, guidance, rng, settings)
        if root.player is None:
            raise ValueError("No legal action to search from")
        player = rbs.players().names[root.player]
        logger.info("Searching %s for %s", _limits(settings), player)
        if possible is not None:
            logger.info("%s sees %d states that could be true", player, len(possible))
        iterations, seconds, depth = self._run(
            settings,
            player,
            lambda iteration: self._iterate(
                iteration, root, rbs, settings, guidance, valuation, rng, possible
            ),
        )
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
        return SearchResult(
            player,
            statistics,
            chosen,
            tuple(self._samples(root, possible is not None)),
            iterations=iterations,
            seconds=seconds,
            depth=depth,
        )

    def _run(self, settings: SearchSettings, player: str, iterate: Callable[[int], int]) -> tuple[int, float, int]:
        """Runs iterations until the settings' iterations are done or their seconds have passed, whichever comes first,
        checking the time between iterations and completing at least one; the iterations done, the seconds taken and the
        most actions from the root an iteration went."""
        started = self._time_source.now()
        deadline = None if settings.seconds is None else Deadline(started + settings.seconds, self._time_source)
        iterations = depth = 0
        while True:
            iterations += 1
            depth = max(depth, iterate(iterations))
            if settings.iterations is not None and iterations >= settings.iterations:
                break
            if deadline is not None and deadline.passed():
                break
        seconds = self._time_source.now() - started
        prior = settings.prior.name if settings.prior is not None else "uniform"
        logger.info(
            "Searched %d iterations in %.3f seconds for %s, %s, tree depth %d",
            iterations,
            seconds,
            player,
            f"puct with the {prior} prior" if settings.selection == PUCT else settings.selection,
            depth,
        )
        return iterations, seconds, depth

    def _iterate(
        self,
        iteration: int,
        root: DecisionNode,
        rbs: RuleBasedGame,
        settings: SearchSettings,
        guidance: Guidance | None,
        valuation: LeafValuation | None,
        rng: random.Random,
        possible: Possible | None,
    ) -> int:
        """One iteration, giving how many actions from the root it went. Given the states the position could be, it
        draws one of them and plays on it: legal actions and
        outcomes come from that state, and a node is what the searching player sees of the states reaching it."""
        node = root
        state = root.state if possible is None else self._draw(possible, rng)
        decisions = [root]
        chances: list[ChanceNode] = []
        while actions := node.actions if possible is None else rbs.actions(state):
            picked: Action | None = None
            if settings.selection == PUCT:
                picked = self._puct(node, actions, settings.puct_exploration)
                action = None if picked in node.children else picked
            else:
                action = self._untried(node, actions, possible is not None, rng)
            if action is not None:
                outcomes = rbs.outcomes(state, action).outcomes
                chance = ChanceNode(action, outcomes, {}, 0, [0.0] * len(rbs.players().names))
                node.children[action] = chance
            elif picked is not None:
                chance = node.children[picked]
                outcomes = (
                    chance.outcomes
                    if possible is None
                    else rbs.outcomes(state, chance.action).outcomes
                )
            else:
                chance = node.children[self._select(node, actions, settings.exploration, guidance)]
                outcomes = (
                    chance.outcomes
                    if possible is None
                    else rbs.outcomes(state, chance.action).outcomes
                )
            chances.append(chance)
            state = self._draw(outcomes, rng)
            node = self._outcome(chance, state, rbs, guidance, rng, possible, settings)
            decisions.append(node)
            if node.visits == 0:
                actions = node.actions
                break

        if not actions:
            payoffs, rollout_length, ending = self._state_reader.payoffs(state, rbs.players()), 0, ""
        else:
            payoffs, rollout_length, ending = self._rollout(
                rbs, state, settings, guidance, valuation, rng
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
                " ".join(f"{name}={payoff}" for name, payoff in zip(rbs.players().names, payoffs)),
            )
        return len(chances)

    def _untried(
        self, node: DecisionNode, actions: tuple[Action, ...], drawn: bool, rng: random.Random
    ) -> Action | None:
        """The next action to try, or None when every legal action has been tried. Under a drawn state, legal actions
        can differ between the states drawn: the untried ones still come in the node's order, and an action the node
        didn't have when created comes in random order after them."""
        if not drawn:
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

    def _puct(self, node: DecisionNode, actions: tuple[Action, ...], exploration: float) -> Action:
        """The action with the highest Q + c · P · √N / (1 + n) among the legal ones, tried or not: Q is the action's mean
        payoff for the player to act, or the node's own mean so far for an action not visited yet, 0 before the node's
        first visit; P its prior, the mean prior for an action the node didn't have when made; N the node's visits and
        n the action's. Ties go to the higher prior, then to the first action."""
        player = node.player
        visited = [chance for chance in node.children.values() if chance.visits]
        visits = sum(chance.visits for chance in visited)
        mean = math.fsum(chance.payoff_sums[player] for chance in visited) / visits if visits else 0.0  # type: ignore[index]
        prior_of = dict(zip(node.actions, node.priors, strict=True))
        fallback = math.fsum(node.priors) / len(node.priors) if node.priors else 1.0 / len(actions)
        root = math.sqrt(node.visits)

        def score(action: Action) -> tuple[float, float]:
            chance = node.children.get(action)
            count = 0 if chance is None else chance.visits
            q = chance.payoff_sums[player] / count if chance is not None and count else mean  # type: ignore[index]
            prior = prior_of.get(action, fallback)
            return q + exploration * prior * root / (1 + count), prior

        return max(actions, key=score)

    def _outcome(
        self,
        chance: ChanceNode,
        state: State,
        rbs: RuleBasedGame,
        guidance: Guidance | None,
        rng: random.Random,
        possible: Possible | None,
        settings: SearchSettings,
    ) -> DecisionNode:
        seen = state
        child = chance.children.get(seen)
        if child is None:
            child = self._decision_node(rbs, state, guidance, rng, settings, seen)
            chance.children[seen] = child
        return child

    def _decision_node(
        self,
        rbs: RuleBasedGame,
        state: State,
        guidance: Guidance | None,
        rng: random.Random,
        settings: SearchSettings,
        seen: State | None = None,
    ) -> DecisionNode:
        """A node for the state, under a `search node` reasoning frame; see `_new_decision_node`."""
        with process_debugger().frame("search node", context=rbs.context, state=state):
            return self._new_decision_node(rbs, state, guidance, rng, settings, seen)

    def _new_decision_node(
        self,
        rbs: RuleBasedGame,
        state: State,
        guidance: Guidance | None,
        rng: random.Random,
        settings: SearchSettings,
        seen: State | None = None,
    ) -> DecisionNode:
        """A node for the state, holding what the searching player sees of it, rated on that, and under PUCT with its
        actions' priors, every action alike without a prior."""
        actions = rbs.actions(state)
        shown = state if seen is None else seen
        ratings = self._ratings(guidance, shown, actions) if actions else ()
        untried = list(actions)
        rng.shuffle(untried)
        if ratings:
            rating_of = dict(zip(actions, ratings, strict=True))
            untried.sort(key=rating_of.__getitem__)
        player = rbs.players().names.index(rbs.acting_player(state)) if actions else None
        priors: tuple[float, ...] = ()
        if settings.selection == PUCT and actions:
            priors = (
                tuple(1.0 / len(actions) for _ in actions)
                if settings.prior is None
                else settings.prior.priors(shown, actions)
            )
        return DecisionNode(shown, actions, untried, {}, 0, player, ratings, priors)

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
        rbs: RuleBasedGame,
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
        while actions := rbs.actions(state):
            if (stopped := self._stopped(state, length, rbs.players(), settings, valuation)) is not None:
                return stopped
            ratings = self._ratings(guidance, state, actions) if guidance is not None and guidance.guided_rollouts else ()
            if guidance is not None and ratings:
                best = max(ratings)
                weights = [math.exp((rating - best) / guidance.rollout_temperature) for rating in ratings]
                (action,) = rng.choices(actions, weights=weights)
            else:
                action = rng.choice(actions)
            outcomes = rbs.outcomes(state, action).outcomes
            state = self._draw(outcomes, rng)
            length += 1
        return self._state_reader.payoffs(state, rbs.players()), length, ""

    def _stopped(
        self,
        state: State,
        length: int,
        players: Players,
        settings: SearchSettings,
        valuation: LeafValuation | None,
    ) -> tuple[tuple[float, ...], int, str] | None:
        """The payoffs ending a rollout still in play early: the valuer's after its rollout actions, unless it knows
        nothing about the position, or the unfinished payoff at the rollout limit; None to play on."""
        if valuation is not None and length == valuation.rollout_actions:
            values = valuation.valuer.values(state)
            if values is not None:
                return values, length, ", then valued"
        if settings.rollout_limit is not None and length >= settings.rollout_limit:
            unfinished = float(settings.unfinished_payoff)  # type: ignore[arg-type]
            return (unfinished,) * len(players.names), length, ", then stopped at the rollout limit"
        return None

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

    def _samples(self, root: DecisionNode, drawn: bool) -> Iterator[ActionSample]:
        """A sample for every expanded action; under drawn states, also those of actions a node didn't have when
        created."""
        pending = [root]
        while pending:
            node = pending.pop()
            if node.player is None:
                continue
            expanded = node.actions
            if drawn:
                expanded = (*node.actions, *(action for action in node.children if action not in node.actions))
            for action in expanded:
                chance = node.children.get(action)
                if chance is None or chance.visits == 0:
                    continue
                mean = chance.payoff_sums[node.player] / chance.visits
                yield ActionSample(node.state, node.player, action, chance.visits, mean)
                pending.extend(chance.children.values())  # type: ignore[arg-type]

    def _search_at_once(
        self,
        rbs: RuleBasedGame,
        state: State,
        settings: SearchSettings,
        valuation: LeafValuation | None,
        player: str | None,
        predicted: Mapping[str, tuple[tuple[Action, float], ...]],
        rng: random.Random,
    ) -> SearchResult:
        """The search from a state where players act at once, for the searching player: regret matching at every node,
        the searching player's action sampled from its average strategy at the root."""
        root = self._node_at_once(rbs, state)
        if not root.actions:
            raise ValueError("No legal action to search from")
        acting = [rbs.players().names[index] for index in root.players]
        if player not in acting:
            raise ValueError(f"A search where {', '.join(acting)} act at once needs one of them as the searching player, not {player!r}")
        seat = root.players.index(rbs.players().names.index(player))  # type: ignore[arg-type]
        root.predicted = self._predicted(root, rbs.players(), seat, predicted)
        logger.info("Searching %s for %s, acting at once with %s", _limits(settings), player, ", ".join(
            name for name in acting if name != player
        ))
        for position, strategy in root.predicted.items():
            logger.info(
                "%s is predicted to play %s",
                acting[position],
                " ".join(
                    f"{self._action_text_mapper.to_text(action)}={probability}"
                    for action, probability in zip(root.actions[position], strategy, strict=True)
                ),
            )
        iterations, seconds, depth = self._run(
            settings,
            player,  # type: ignore[arg-type]
            lambda iteration: self._iterate_at_once(iteration, root, rbs, settings, valuation, rng),
        )
        actions, counts, sums = root.actions[seat], root.counts[seat], root.strategy_sums[seat]
        statistics = tuple(
            ActionStatistics(action, counts[index], root.payoff_sums[seat][index] / counts[index] if counts[index] else 0.0)
            for index, action in enumerate(actions)
        )
        total = math.fsum(sums)
        strategy = tuple(
            (action, sums[index] / total if total > 0.0 else 1.0 / len(actions)) for index, action in enumerate(actions)
        )
        (chosen,) = rng.choices(actions, weights=[probability for _, probability in strategy])
        for item, (_, probability) in zip(statistics, strategy, strict=True):
            logger.info(
                "%s: %d visits, mean payoff %s, average strategy %s for %s",
                self._action_text_mapper.to_text(item.action),
                item.visits,
                item.mean_payoff,
                probability,
                player,
            )
        logger.info("Sampled from the average strategy: %s", self._action_text_mapper.to_text(chosen))
        return SearchResult(
            player,  # type: ignore[arg-type]
            statistics,
            chosen,
            tuple(self._samples_at_once(root)),
            strategy,
            iterations,
            seconds,
            depth=depth,
        )

    def _predicted(
        self,
        root: SimultaneousNode,
        players: Players,
        seat: int,
        predicted: Mapping[str, tuple[tuple[Action, float], ...]],
    ) -> dict[int, tuple[float, ...]]:
        """The predicted strategies by position at the root, over each player's legal actions in order. A player not to
        act, the searching player itself, an action it can't take, a negative probability, or probabilities not summing
        to 1 raise ValueError."""
        strategies: dict[int, tuple[float, ...]] = {}
        for name, mix in predicted.items():
            index = players.names.index(name) if name in players.names else None
            if index is None or index not in root.players:
                raise ValueError(f"{name!r} isn't a player to act, so its strategy can't be predicted")
            position = root.players.index(index)
            if position == seat:
                raise ValueError(f"{name} is the searching player: its own strategy can't be predicted")
            weights = dict(mix)
            if unknown := [action for action in weights if action not in root.actions[position]]:
                raise ValueError(f"{name} can't take {self._action_text_mapper.to_text(unknown[0])}")
            probabilities = tuple(float(weights.get(action, 0.0)) for action in root.actions[position])
            total = math.fsum(probabilities)
            if min(probabilities) < 0.0 or not math.isclose(total, 1.0):
                raise ValueError(f"{name}'s predicted strategy needs probabilities from 0 summing to 1, not {probabilities}")
            strategies[position] = probabilities
        return strategies

    def _iterate_at_once(
        self,
        iteration: int,
        root: SimultaneousNode,
        rbs: RuleBasedGame,
        settings: SearchSettings,
        valuation: LeafValuation | None,
        rng: random.Random,
    ) -> int:
        """One iteration where players act at once, giving how many joint actions from the root it went: every player to act picks by regret matching, the joint action's
        outcome is drawn, and from the first new node a rollout of uniformly drawn joint actions plays on; then each
        node's regrets, strategies and statistics take the payoffs."""
        node, state = root, root.state
        visited = [root]
        chances: list[ChanceNode] = []
        picks: list[tuple[SimultaneousNode, Picked]] = []
        while node.actions:
            picked = self._pick(node, settings.regret_exploration, rng)
            joint = JointAction(
                tuple(
                    (rbs.players().names[node.players[position]], node.actions[position][index])
                    for position, (index, _) in enumerate(picked)
                )
            )
            chance = node.children.get(joint)
            if chance is None:
                outcomes = rbs.joint_outcomes(state, joint).outcomes
                chance = ChanceNode(joint, outcomes, {}, 0, [0.0] * len(rbs.players().names))
                node.children[joint] = chance
            picks.append((node, picked))
            chances.append(chance)
            state = self._draw(chance.outcomes, rng)
            child = chance.children.get(state)
            if child is None:
                child = self._node_at_once(rbs, state)
                chance.children[state] = child
            visited.append(child)  # type: ignore[arg-type]
            node = child  # type: ignore[assignment]
            if node.visits == 0:
                break

        if not node.actions:
            payoffs, rollout_length, ending = self._state_reader.payoffs(state, rbs.players()), 0, ""
        else:
            payoffs, rollout_length, ending = self._rollout_at_once(rbs, state, settings, valuation, rng)

        for item in visited:
            item.visits += 1
        for chance in chances:
            chance.visits += 1
            for index, payoff in enumerate(payoffs):
                chance.payoff_sums[index] += payoff
        for picked_node, picked in picks:
            self._learn(picked_node, picked, payoffs)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Iteration %d: %s, rollout of %d actions%s, payoffs %s",
                iteration,
                " > ".join(self._action_text_mapper.joint_text(chance.action) for chance in chances),  # type: ignore[arg-type]
                rollout_length,
                ending,
                " ".join(f"{name}={payoff}" for name, payoff in zip(rbs.players().names, payoffs)),
            )
        return len(chances)

    def _node_at_once(self, rbs: RuleBasedGame, state: State) -> SimultaneousNode:
        legal = rbs.joint_actions(state)
        actions = tuple(actions for _, actions in legal)
        return SimultaneousNode(
            state,
            tuple(index for index, _ in legal),
            actions,
            [[0.0] * len(choices) for choices in actions],
            [[0.0] * len(choices) for choices in actions],
            [[0.0] * len(choices) for choices in actions],
            [[0] * len(choices) for choices in actions],
            {},
            0,
        )

    def _pick(self, node: SimultaneousNode, exploration: float, rng: random.Random) -> Picked:
        """Each player to act draws an action: a predicted player from its predicted strategy, any other from its
        regret-matching strategy mixed with uniform choice, that strategy adding to its summed strategies."""
        picked: list[tuple[int, float]] = []
        for position, choices in enumerate(node.actions):
            count = len(choices)
            mixed = node.predicted.get(position)
            if mixed is None:
                strategy = self._regret_matching(node.regrets[position])
                for index, probability in enumerate(strategy):
                    node.strategy_sums[position][index] += probability
                mixed = tuple((1.0 - exploration) * probability + exploration / count for probability in strategy)
            (index,) = rng.choices(range(count), weights=mixed)
            picked.append((index, mixed[index]))
        return tuple(picked)

    def _regret_matching(self, regrets: list[float]) -> tuple[float, ...]:
        """Each action's probability in proportion to its positive regret; uniform when no regret is positive."""
        positive = [max(regret, 0.0) for regret in regrets]
        total = math.fsum(positive)
        if total <= 0.0:
            return (1.0 / len(regrets),) * len(regrets)
        return tuple(regret / total for regret in positive)

    def _learn(self, node: SimultaneousNode, picked: Picked, payoffs: tuple[float, ...]) -> None:
        """Adds each player's payoff to its picked action's statistics and, for a player not predicted, updates its
        regrets with the outcome-sampling estimate: the picked action's payoff divided by the probability it was picked
        at, every action's regret growing by its estimate minus the payoff."""
        for position, (index, probability) in enumerate(picked):
            payoff = payoffs[node.players[position]]
            node.counts[position][index] += 1
            node.payoff_sums[position][index] += payoff
            if position in node.predicted:
                continue
            regrets = node.regrets[position]
            for action in range(len(regrets)):
                regrets[action] += (payoff / probability if action == index else 0.0) - payoff

    def _rollout_at_once(
        self,
        rbs: RuleBasedGame,
        state: State,
        settings: SearchSettings,
        valuation: LeafValuation | None,
        rng: random.Random,
    ) -> tuple[tuple[float, ...], int, str]:
        """A rollout where players act at once: every player to act draws uniformly among its legal actions."""
        length = 0
        while legal := rbs.joint_actions(state):
            if (stopped := self._stopped(state, length, rbs.players(), settings, valuation)) is not None:
                return stopped
            joint = JointAction(tuple((rbs.players().names[index], rng.choice(actions)) for index, actions in legal))
            state = self._draw(rbs.joint_outcomes(state, joint).outcomes, rng)
            length += 1
        return self._state_reader.payoffs(state, rbs.players()), length, ""

    def _samples_at_once(self, root: SimultaneousNode) -> Iterator[ActionSample]:
        """A sample for every action a player to act took anywhere in the tree, with its visits and mean payoff for
        that player."""
        pending = [root]
        while pending:
            node = pending.pop()
            for position, index in enumerate(node.players):
                for choice, action in enumerate(node.actions[position]):
                    count = node.counts[position][choice]
                    if count:
                        yield ActionSample(node.state, index, action, count, node.payoff_sums[position][choice] / count)
            for chance in node.children.values():
                pending.extend(chance.children.values())  # type: ignore[arg-type]


def _limits(settings: SearchSettings) -> str:
    """What stops a search, as its log line says it: `100 iterations`, `for 2.5 seconds`, or `up to 100 iterations or
    2.5 seconds`."""
    if settings.seconds is None:
        return f"{settings.iterations} iterations"
    if settings.iterations is None:
        return f"for {settings.seconds:g} seconds"
    return f"up to {settings.iterations} iterations or {settings.seconds:g} seconds"
