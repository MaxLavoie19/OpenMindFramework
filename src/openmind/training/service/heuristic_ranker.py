import logging
import math
import random
from dataclasses import replace

from openmind.agent.service.outfitter import Outfitter
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.model.guidance import Guidance
from openmind.training.model.arm import Arm, ArmScore
from openmind.training.model.ranking_settings import RankingSettings
from openmind.training.service.arm_selector import ArmSelector
from openmind.training.service.self_play import SelfPlay

logger = logging.getLogger(__name__)


class HeuristicRanker:
    """Ranks heuristics by playing them against each other, choosing what to play next by UCB1.

    A heuristic is worth what playing with it is worth, which no fit can say: a term can explain every position a game
    went through and still not help anyone win. So each is an arm, a pull is a game against another heuristic, and
    what it paid is the reward.

    They are played in pairs because a heuristic playing itself wins half its games whatever it is. Which pair comes
    next is UCB1's: a heuristic that lost its first game may have been unlucky, and how long to keep trying it is
    exactly the question a bandit answers."""

    def __init__(self, self_play: SelfPlay, arm_selector: ArmSelector, outfitter: Outfitter) -> None:
        self._self_play = self_play
        self._selector = arm_selector
        self._outfitter = outfitter

    def rank(
        self, knowledge_base: KnowledgeBase, game: RuleBasedGame, arms: tuple[Arm, ...], settings: RankingSettings
    ) -> tuple[ArmScore, ...]:
        """Plays the games the settings allow and gives every heuristic with what its games came to, best first.
        Fewer than two heuristics raise ValueError: there is nothing to rank one against."""
        if len(arms) < 2:
            raise ValueError(f"Ranking heuristics needs at least two, not {len(arms)}")
        rng = random.Random(settings.seed)
        scores: dict[str, tuple[int, float]] = {}
        players = game.players().names
        for number in range(settings.games):
            first, second = self._selector.pair(scores, {}, [arm.name for arm in arms], settings.exploration, rng)
            playing = {name: arm for arm in arms for name in (arm.name,)}
            guidance = self._guidance(knowledge_base, players, (playing[first], playing[second]))
            (played,) = self._self_play.play(
                knowledge_base, game, guidance, replace(settings.play, games=1, seed=settings.seed + number)
            )
            self._scored(scores, (first, second), played.payoffs)
            logger.info(
                "Game %d of %d: %s against %s paid %s",
                number + 1,
                settings.games,
                first,
                second,
                played.payoffs or "nobody",
            )
        return self._ranked(arms, scores, settings.exploration)

    def _guidance(
        self, knowledge_base: KnowledgeBase, players: tuple[str, ...], playing: tuple[Arm, Arm]
    ) -> dict[str, Guidance]:
        """What each player plays with: the heuristic of the arm it stands for, loaded as the search asks for it. A
        game of more than two players gives the rest the first arm, since a pair is what is being told apart."""
        filled = [self._outfitter.filled(knowledge_base, arm.model) for arm in playing]
        return {
            player: Guidance(player, filled[number] if number < len(filled) else filled[0])
            for number, player in enumerate(players)
        }

    def _scored(self, scores: dict[str, tuple[int, float]], playing: tuple[str, str], payoffs: tuple[float, ...]) -> None:
        """What the game paid each side, added to what they had. A game that paid nobody counts for neither."""
        if len(payoffs) < 2:
            return
        for number, arm in enumerate(playing):
            games, points = scores.get(arm, (0, 0.0))
            scores[arm] = (games + 1, points + payoffs[number])

    def _ranked(
        self, arms: tuple[Arm, ...], scores: dict[str, tuple[int, float]], exploration: float
    ) -> tuple[ArmScore, ...]:
        """Every heuristic with what its games came to, best first; a heuristic that never played is worth nothing
        known and goes last."""
        total = sum(games for games, _ in scores.values())
        ranked = []
        for arm in arms:
            games, points = scores.get(arm.name, (0, 0.0))
            bound = (
                None
                if not games
                else points / games + exploration * math.sqrt(math.log(max(total, 1)) / games)
            )
            ranked.append(ArmScore(arm, games, points, bound))
        return tuple(sorted(ranked, key=lambda score: (0 if score.games else 1, -score.mean, -score.games)))
