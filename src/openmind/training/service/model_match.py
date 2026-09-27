import logging
from dataclasses import replace

from openmind.agent.service.outfitter import Outfitter
from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.knowledge.constant.task_constant import MOVE_VALUE, POSITION_VALUE
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.knowledge.model.model_record import ModelRecord
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.model.guidance import Guidance
from openmind.training.model.match import Match
from openmind.training.model.self_play_settings import SelfPlaySettings
from openmind.training.service.self_play import SelfPlay

logger = logging.getLogger(__name__)


class ModelMatch:
    """Plays two models of the same task against each other and writes down what came of it.

    **Games are the evidence and the registry is the selector, and this is the link between them that was
    missing.** Every finished game was already kept — as a direct experience, with a payoff and an outcome
    belief per player, each tagged with the model that played that side. None of it reached the one belief
    anything selects on, so a model could win a hundred games and `best` would go on preferring whichever was
    registered first. The games were evidence nothing read.

    **A model is measured by the share of the points it took**, a win one, a draw a half, a loss none. That is
    a measure of *winning*, not of being well calibrated, and the difference is deliberate: how close a value
    is to the truth is what the held-out loss of the fit already measures, and a heuristic can be beautifully
    calibrated and still lose. What a match adds is the only thing the fit cannot say.

    **Each plays each side.** A game where one model always moves first measures the first move as much as the
    model, so the games are split between the two seatings and the result says how they were split. Nothing
    here corrects for an odd number; it reports it and leaves the reading to whoever reads it.

    Nothing about any game is known here. Which player goes first, whether a draw is possible, how many
    players there are — all of it comes from the game, and a game of more than two players is declined rather
    than guessed at, because a match of two is what this is."""

    def __init__(
        self,
        self_play: SelfPlay,
        outfitter: Outfitter,
        accuracy_scorer: AccuracyScorer | None = None,
    ) -> None:
        self._self_play = self_play
        self._outfitter = outfitter
        self._scorer = AccuracyScorer() if accuracy_scorer is None else accuracy_scorer

    def play(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        one: ModelRecord,
        other: ModelRecord,
        settings: SelfPlaySettings,
    ) -> Match | None:
        """Those two models over that many games, half each way, scored onto both.

        None where the two are not models of one task, where the game is not of two players, or where either
        model cannot be loaded — a match nothing could play is no result, not a nil-all draw."""
        task = self._shared(one, other)
        players = game.players().names
        if task is None or len(players) != 2:
            logger.info(
                "No match between %s and %s: %s",
                one.name,
                other.name,
                "they are models of no one task" if task is None else f"{len(players)} players, and a match is of two",
            )
            return None
        filled = (self._outfitter.filled(knowledge_base, one), self._outfitter.filled(knowledge_base, other))
        if filled[0] is None or filled[1] is None:
            logger.info("No match between %s and %s: one of them would not load", one.name, other.name)
            return None
        points, sides, played = [0.0, 0.0], [0, 0], 0
        for seating in (0, 1):
            games = settings.games // 2 + (settings.games % 2 if seating == 0 else 0)
            if games < 1:
                continue
            first = seating
            # Which model sits as which player, swapped on the second seating so that neither is measured on
            # having the first move.
            at = (seating, 1 - seating)
            guidance = {
                player: self._guided(player, task, filled[at[index]])
                for index, player in enumerate(players)
            }
            for result in self._self_play.play(
                knowledge_base, game, guidance, replace(settings, games=games, seed=settings.seed + seating * 1000)
            ):
                if len(result.payoffs) != len(players):
                    continue
                played += 1
                sides[first] += 1
                for index in range(2):
                    points[at[index]] += self._points(result.payoffs, index)
        if not played:
            logger.info("No match between %s and %s: no game of theirs finished", one.name, other.name)
            return None
        # Both played every game, one on each side of it, so both are scored over the same count.
        for model, took in ((one, points[0]), (other, points[1])):
            self._scorer.played(knowledge_base, model.mechanism, model.context, took, played)
        match = Match(game.context, task, one.name, other.name, played, (points[0], points[1]), (sides[0], sides[1]))
        logger.info(
            "%s took %.4g and %s took %.4g of %d games of %s at %s",
            one.name,
            match.points[0],
            other.name,
            match.points[1],
            played,
            game.context,
            task,
        )
        return match

    def _shared(self, one: ModelRecord, other: ModelRecord) -> str | None:
        """The heuristic task both are models of, or None where there is none.

        Two models are comparable by playing only where they answer the same question. A position value and a
        move value both play a game and what they did is not a comparison of anything."""
        for task in (POSITION_VALUE, MOVE_VALUE):
            if task in one.tasks and task in other.tasks:
                return task
        return None

    def _guided(self, player: str, task: str, filled: tuple[object, object]) -> Guidance:
        """That player guided by that model, in the slot its task belongs to."""
        if task == POSITION_VALUE:
            return Guidance(player, position_value=filled)  # type: ignore[arg-type]
        return Guidance(player, move_value=filled)  # type: ignore[arg-type]

    def _points(self, payoffs: tuple[float, ...], index: int) -> float:
        """What that seat took of the one point a game is worth: all of it for coming out ahead of the other,
        half for level, none for behind.

        Read off what the game paid rather than off any notion of winning, so a game that pays in something
        other than nought and one is scored the same way. What matters is which player the game paid more."""
        mine, theirs = payoffs[index], payoffs[1 - index]
        if mine > theirs:
            return 1.0
        return 0.5 if mine == theirs else 0.0
