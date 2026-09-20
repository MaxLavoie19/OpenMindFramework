import logging
import math
import random
from collections.abc import Sequence

from openmind.heuristic.model.node import Node
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.search.model.guidance import Guidance
from openmind.search.model.search_node import ActionStatistics, SearchNode
from openmind.search.model.search_settings import SearchSettings
from openmind.search.model.strategy import MoveDistribution, Strategy
from openmind.search.repository.search_tree_cache import SearchTreeCache
from openmind.utility.service.utility import Utility
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction

logger = logging.getLogger(__name__)


class MonteCarloTreeSearch:
    """The planner that explores the lines a position leads to, as many nodes as it is given, and gives back the
    strategy it settled on.

    Every iteration goes down the tree to a position it hasn't expanded, expands it, reads what it is worth, and
    carries that back up the way it came. A position the game is over in is worth what the game paid; any other is
    worth what the position value heuristic says. **Nothing is played out to the end**: a search without a heuristic
    only learns from the wins it happens to reach, which is enough in a small game and nothing like enough in a large
    one — that is what heuristics inferred from the rules are for.

    There are no turns in OMF, so there is one search rather than two: where one player acts, its action is picked by
    PUCT, what its model rated the move weighed against what the visits found; where several act at once, each picks
    by regret matching over its own actions, and what the node settles on is the average of those strategies.

    It keeps nothing: the tree lives in the cache it is given, so the next move carries on from what this one found."""

    def __init__(self, utility: Utility, trees: SearchTreeCache | None = None) -> None:
        self._utility = utility
        self._trees = SearchTreeCache() if trees is None else trees

    def plan(
        self,
        model: object,
        knowledge_base: KnowledgeBase,
        node: Node,
        guidance: Guidance,
        settings: SearchSettings,
    ) -> Strategy | None:
        """The strategy worked out from this position; None where nothing could be worked out. A search told nothing
        about how much to explore raises ValueError: it would never stop."""
        if settings.nodes is None:
            raise ValueError("A Monte-Carlo tree search needs to be told how many nodes it may explore")
        players = node.game.players().names  # type: ignore[union-attr]
        root = self._trees.tree(node.state) or SearchNode(node)
        rng = random.Random(settings.seed)
        valued = sum(self._iterate(root, guidance, settings, players, rng) for _ in range(settings.nodes))
        self._trees.keep(node.state, root)
        moves = self._moves(root, guidance.player, settings.temperature)
        if valued == 0:
            logger.info(
                "Explored %d nodes of %s without valuing one: nothing paid out and no heuristic said what a position "
                "is worth",
                settings.nodes,
                node.game.context,  # type: ignore[union-attr]
            )
        if not moves:
            return None
        logger.info(
            "Explored %s for %s: %d nodes, %d states covered", node.game.context, guidance.player, root.visits, len(moves)  # type: ignore[union-attr]
        )
        return Strategy(tuple(moves.items()))

    def _iterate(
        self, root: SearchNode, guidance: Guidance, settings: SearchSettings, players: Sequence[str], rng: random.Random
    ) -> bool:
        """One iteration, from the root down to a position to read and back up again; whether anything was valued.

        A position that could not be valued still counts as a visit, so the search moves on to another line rather
        than going down the same one forever."""
        path: list[tuple[SearchNode, JointAction]] = []
        current = root
        depth = 0
        while True:
            if not current.expanded:
                self._expand(current, guidance, players)
            if current.over:
                values = self._paid(current, players)
                break
            if current.visits == 0 or (settings.depth is not None and depth >= settings.depth):
                values = self._valued(current, guidance, players)
                break
            joint = self._select(current, settings, players, rng)
            following = self._follow(current, joint, rng)
            if following is None:
                values = self._valued(current, guidance, players)
                break
            path.append((current, joint))
            current, depth = following, depth + 1
        self._back_up(path, current, values, players)
        return values is not None

    def _expand(self, node: SearchNode, guidance: Guidance, players: Sequence[str]) -> None:
        """The actions every player has here, each with what its model rates it: what the search picks among from now
        on. A position nobody can act in is left with none, which is what says the game is over there."""
        game = node.node.game
        for index, actions in game.joint_actions(node.node.state):  # type: ignore[union-attr]
            player = players[index]
            node.actions[player] = tuple(
                ActionStatistics(action, prior)
                for action, prior in zip(actions, self._priors(node.node, actions, player, guidance), strict=True)
            )
        node.expanded = True

    def _priors(
        self, node: Node, actions: tuple[Action, ...], player: str, guidance: Guidance
    ) -> tuple[float, ...]:
        """What each action is rated by the model of the player taking it — the agent's own move value model for the
        player it plans for, that agent's model for anyone else — as a distribution over the actions. Uniform where no
        model rates them, and where a model rates none of them."""
        filled = guidance.move_value if player == guidance.player else guidance.agents.get(player)
        uniform = (1.0 / len(actions),) * len(actions)
        if filled is None:
            return uniform
        model, rater = filled
        rated = rater.rate(model, node, actions, player)
        known = [value for value in rated if value is not None]
        if not known:
            return uniform
        floor = min(known)
        weights = [math.exp(float(value if value is not None else floor) - max(known)) for value in rated]
        total = math.fsum(weights)
        return tuple(weight / total for weight in weights) if total else uniform

    def _select(
        self, node: SearchNode, settings: SearchSettings, players: Sequence[str], rng: random.Random
    ) -> JointAction:
        """What the acting players do here: one player picks by PUCT, several by regret matching, each over its own
        actions."""
        alone = len(node.actions) == 1
        picked: list[tuple[str, Action]] = []
        for player in players:
            statistics = node.actions.get(player)
            if not statistics:
                continue
            chosen = (
                self._by_confidence(node, statistics, settings)
                if alone
                else self._by_regret(node, statistics, settings, players.index(player), rng)
            )
            picked.append((player, chosen.action))
        return JointAction(tuple(picked))

    def _by_confidence(
        self, node: SearchNode, statistics: Sequence[ActionStatistics], settings: SearchSettings
    ) -> ActionStatistics:
        """PUCT: what taking the action has brought, plus what it was rated, weighed against how little it has been
        tried. A move nothing has tried is taken before one that has."""
        visits = max(node.visits, 1)

        def score(statistic: ActionStatistics) -> float:
            return statistic.mean + settings.exploration * statistic.prior * math.sqrt(visits) / (1 + statistic.visits)

        return max(statistics, key=score)

    def _by_regret(
        self,
        node: SearchNode,
        statistics: Sequence[ActionStatistics],
        settings: SearchSettings,
        player: int,
        rng: random.Random,
    ) -> ActionStatistics:
        """Regret matching: each action as likely as what not taking it has cost, mixed with a uniform choice so a
        move nothing regrets yet is still tried. What it settles on is summed, so the average strategy can be read off
        it. Every action is tried once before any regret is read."""
        probabilities = self._regret_matching(node, statistics, player)
        for statistic, probability in zip(statistics, probabilities, strict=True):
            statistic.strategy += probability
        uniform = 1.0 / len(statistics)
        mixed = [
            (1.0 - settings.regret_exploration) * probability + settings.regret_exploration * uniform
            for probability in probabilities
        ]
        return rng.choices(list(statistics), weights=mixed)[0]

    def _regret_matching(
        self, node: SearchNode, statistics: Sequence[ActionStatistics], player: int
    ) -> tuple[float, ...]:
        """Each action in proportion to what not taking it has cost the player: how much better it has done than this
        position has. Uniform over what hasn't been tried while anything hasn't, and uniform where nothing regrets."""
        untried = [1.0 if statistic.visits == 0 else 0.0 for statistic in statistics]
        if any(untried):
            return tuple(value / math.fsum(untried) for value in untried)
        here = node.mean(player)
        regrets = [max(statistic.mean - here, 0.0) for statistic in statistics]
        total = math.fsum(regrets)
        if total <= 0.0:
            return (1.0 / len(statistics),) * len(statistics)
        return tuple(regret / total for regret in regrets)

    def _follow(self, node: SearchNode, joint: JointAction, rng: random.Random) -> SearchNode | None:
        """Where the joint action leads: one of its outcomes, drawn by how likely the predictor says it is, kept as a
        child of its own so a game of chance branches as it does. None where it leads nowhere."""
        outcomes = node.node.game.joint_outcomes(node.node.state, joint).outcomes  # type: ignore[union-attr]
        if not outcomes:
            return None
        drawn = rng.choices([state for state, _ in outcomes], weights=[chance for _, chance in outcomes])[0]
        key = (joint, drawn)
        child = node.children.get(key)
        if child is None:
            child = node.children[key] = SearchNode(node.node.of(drawn))
        return child

    def _back_up(
        self,
        path: Sequence[tuple[SearchNode, JointAction]],
        leaf: SearchNode,
        values: tuple[float, ...] | None,
        players: Sequence[str],
    ) -> None:
        """What the leaf was worth, carried back up the way the iteration came: every position visited, and every
        action taken along it credited to the player who took it."""
        leaf.visits += 1
        leaf.values = self._added(leaf.values, values, players)
        for node, joint in path:
            node.visits += 1
            node.values = self._added(node.values, values, players)
            for player, action in joint.actions:
                statistic = node.statistics(player, action)
                if statistic is None:
                    continue
                statistic.visits += 1
                if values is not None:
                    statistic.value += values[players.index(player)]

    def _added(
        self, held: tuple[float, ...], values: tuple[float, ...] | None, players: Sequence[str]
    ) -> tuple[float, ...]:
        if values is None:
            return held or (0.0,) * len(players)
        kept = held or (0.0,) * len(players)
        return tuple(one + other for one, other in zip(kept, values, strict=True))

    def _moves(self, root: SearchNode, player: str, temperature: float) -> dict[object, MoveDistribution]:
        """What to play in every position the search explored: each of the player's actions with how often it settled
        on it.

        A move it never explored isn't in the distribution at all: exploring is what says a move is worth considering,
        and a move it didn't think worth a single look isn't one to recommend. A move tried once while others were
        tried dozens of times carries almost none of the distribution, so it is never the one played either."""
        moves: dict[object, MoveDistribution] = {}
        for node in self._explored(root):
            statistics = node.actions.get(player)
            if not statistics:
                continue
            distribution = self._distribution(node, statistics, temperature)
            if distribution:
                moves[node.node.state] = distribution
        return moves

    def _distribution(
        self, node: SearchNode, statistics: Sequence[ActionStatistics], temperature: float
    ) -> MoveDistribution:
        """What the search settled on: how often each action was taken where one player acts, and the average of the
        strategies it played where several did. What it never explored is left out.

        Exploring a move isn't the same as playing it, and how much of what it explored a search plays is the
        temperature's: at 1 it plays what it explored, at 0 the best of it alone, and in between it leans toward the
        best without dropping the rest."""
        weights = [
            (statistic, statistic.strategy if len(node.actions) > 1 else float(statistic.visits))
            for statistic in statistics
        ]
        explored = [(statistic, weight) for statistic, weight in weights if weight > 0.0]
        if not explored:
            return ()
        best = max(weight for _, weight in explored)
        if temperature <= 0.0:
            top = [statistic for statistic, weight in explored if weight == best]
            return tuple((statistic.action, 1.0 / len(top)) for statistic in top)
        sharpened = [(statistic, (weight / best) ** (1.0 / temperature)) for statistic, weight in explored]
        total = math.fsum(weight for _, weight in sharpened)
        if total <= 0.0:
            return ()
        return tuple((statistic.action, weight / total) for statistic, weight in sharpened)

    def _explored(self, root: SearchNode) -> list[SearchNode]:
        """Every position the search reached, the root first."""
        found: list[SearchNode] = []
        pending = [root]
        seen: set[int] = set()
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            found.append(node)
            pending.extend(node.children.values())
        return found

    def _paid(self, node: SearchNode, players: Sequence[str]) -> tuple[float, ...] | None:
        """What the finished position paid each player: a win always carries the payoff value."""
        game = node.node.game
        payoff = game.players().payoff  # type: ignore[union-attr]
        state = node.node.state
        held = state.model(payoff) if state.has(payoff) else None
        paid: list[float] = []
        for player in players:
            value = held.get(player) if held is not None and hasattr(held, "get") else None
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            paid.append(float(value))
        return tuple(paid)

    def _valued(self, node: SearchNode, guidance: Guidance, players: Sequence[str]) -> tuple[float, ...] | None:
        """What the position value heuristic says the position is worth to each player; None where the caller gave no
        model, which is where a search learns only from the wins it reaches."""
        filled = guidance.position_value
        if filled is None:
            return None
        model, valuer = filled
        values = valuer.values(model, node.node)
        return None if values is None or len(values) != len(players) else tuple(values)
