import logging
import time
from dataclasses import replace

from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.model.heuristic_target import HeuristicTarget
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.search.model.guidance import Guidance
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.model.distillation import Distillation
from openmind.training.model.self_play_settings import SelfPlaySettings
from openmind.training.service.self_play import SelfPlay

logger = logging.getLogger(__name__)


class ValueDistiller:
    """Learns a position heuristic from games the agent played against itself.

    Pondering the rules gives a game its first heuristics where the rules settle enough positions; where they don't —
    chess, where nothing is proved and nothing is paid until a hundred moves in — games are what is left. Every
    position a game went through is valued at what that game paid, the candidates the readings allow are composed and
    fitted on those, and what holds up on games it was not fitted on is declared and registered as a model.

    It says nothing about whether the heuristic is any good. What it gives back is what it rests on: how many games,
    how many of them anyone won, and what the fit was off by on games it never saw."""

    def __init__(
        self, self_play: SelfPlay, position_row_mapper: PositionRowMapper, value_generator: ValueGenerator
    ) -> None:
        self._self_play = self_play
        self._rows = position_row_mapper
        self._generator = value_generator

    def distill(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        guidance: Guidance,
        play: SelfPlaySettings,
        values: ValueSettings,
        held_out_games: int = 0,
    ) -> Distillation:
        """Plays, fits, and declares what held up. `held_out_games` are played after the rest and kept back, so the
        rules are chosen on games they were not fitted on."""
        started = time.monotonic()
        played = self._self_play.play(knowledge_base, game, guidance, play)
        held_out = (
            self._self_play.play(
                knowledge_base, game, guidance, replace(play, games=held_out_games, seed=play.seed + play.games)
            )
            if held_out_games
            else ()
        )
        training = self._rows.to_rows(game, played)
        kept_back = self._rows.to_rows(game, held_out)
        decisive = sum(1 for one in (*played, *held_out) if one.decisive)
        if not training:
            logger.info("Nothing to learn from %d games of %s: none of them paid anyone", len(played), game.context)
            return Distillation(game.context, (), len(played), decisive, 0, 0, time.monotonic() - started)
        generated = self._generator.generate(
            game, training, kept_back, values, HeuristicTarget(knowledge_base, game.context)
        )
        logger.info(
            "Distilled %d rules of %s from %d games, %d of them decisive: %d rows, %d held back",
            len(generated.rules),
            game.context,
            len(played) + len(held_out),
            decisive,
            len(training),
            len(kept_back),
        )
        return Distillation(
            generated.context,
            generated.rules,
            len(played) + len(held_out),
            decisive,
            len(training),
            len(kept_back),
            time.monotonic() - started,
            generated.chosen.held_out_loss if generated.chosen is not None else None,
        )
