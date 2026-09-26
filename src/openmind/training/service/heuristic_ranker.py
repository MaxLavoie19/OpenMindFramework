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
        Fewer than two heuristics raise ValueError: there is nothing to rank one against.

        **With no number of games it plays until something stops it**, writing the standings out as it goes. A
        bandit has no point at which it is finished — it is a way of spending attention, not a procedure that
        terminates — so the honest shape is a run that reports and keeps going. Stopped from outside, what it
        has learned so far is returned rather than lost."""
        if len(arms) < 2:
            raise ValueError(f"Ranking heuristics needs at least two, not {len(arms)}")
        rng = random.Random(settings.seed)
        scores: dict[str, tuple[int, float]] = {}
        players = game.players().names
        playing = {arm.name: arm for arm in arms}
        number = 0
        try:
            while settings.games is None or number < settings.games:
                first, second = self._selector.pair(scores, {}, [arm.name for arm in arms], settings.exploration, rng)
                guidance = self._guidance(knowledge_base, players, (playing[first], playing[second]))
                (played,) = self._self_play.play(
                    knowledge_base, game, guidance, replace(settings.play, games=1, seed=settings.seed + number)
                )
                self._scored(scores, (first, second), played.payoffs)
                number += 1
                logger.info(
                    "Game %d of %s: %s against %s paid %s",
                    number,
                    "no end" if settings.games is None else settings.games,
                    first,
                    second,
                    played.payoffs or "nobody",
                )
                if settings.standings and number % settings.standings == 0:
                    self._standings(number, self._ranked(arms, scores, settings.exploration))
        except KeyboardInterrupt:
            logger.info("Stopped after %d games", number)
        return self._ranked(arms, scores, settings.exploration)

    def _standings(self, played: int, ranked: tuple[ArmScore, ...]) -> None:
        """The table so far, headed, so a run with no end can be read while it runs."""
        logger.info("After %d games: %-40s %7s %7s %7s %7s", played, "heuristic", "games", "points", "per game", "bound")
        for score in ranked:
            logger.info(
                "                  %-40s %7d %7.1f %7.3f %7s",
                score.arm.name,
                score.games,
                score.points,
                score.mean,
                "-" if score.bound is None else f"{score.bound:.3f}",
            )

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
