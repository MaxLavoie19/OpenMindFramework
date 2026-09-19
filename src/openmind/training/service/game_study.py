from collections.abc import Sequence

from openmind.debug.factory.debugger_factory import process_debugger
from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.training.model.continuous_training_settings import ContinuousTrainingSettings
from openmind.training.model.game_lesson import GameLesson
from openmind.training.service.ending_walker import EndingWalker
from openmind.training.service.self_play import SelfPlay


class GameStudy:
    """What a worker does with one call of continuous training: plays a game between two arms, then, for a decisive game
    when the settings say so, walks it back from its end before taking the next one. Its services work in this process."""

    def __init__(self, self_play: SelfPlay, ending_walker: EndingWalker) -> None:
        self._self_play = self_play
        self._ending_walker = ending_walker

    def play_and_study(
        self,
        rbs: RuleBasedSystem,
        builders: Sequence[AgentBuilder],
        arms: tuple[str, ...],
        agent_seed: int,
        outcome_seed: int,
        settings: ContinuousTrainingSettings,
    ) -> GameLesson:
        with process_debugger().frame("task", context=rbs.context, details={"task": "play and study", "arms": ", ".join(arms)}):
            return self._play_and_study(rbs, builders, arms, agent_seed, outcome_seed, settings)

    def _play_and_study(
        self,
        rbs: RuleBasedSystem,
        builders: Sequence[AgentBuilder],
        arms: tuple[str, ...],
        agent_seed: int,
        outcome_seed: int,
        settings: ContinuousTrainingSettings,
    ) -> GameLesson:
        game = self._self_play.play_arm_game(
            rbs, builders, arms, agent_seed, outcome_seed, keep_samples=False, time_control=settings.time_control
        )
        if settings.deduction is None or settings.ponder_endings <= 0 or len(set(game.payoffs)) < 2:
            return GameLesson(game)
        return GameLesson(game, self._ending_walker.walk_back(rbs, game.states, settings.deduction, settings.ponder_endings))
