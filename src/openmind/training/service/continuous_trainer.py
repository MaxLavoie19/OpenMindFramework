import logging
import random
from collections.abc import Callable
from dataclasses import replace

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.model.domain import Domain
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.service.game_memory import GameMemory
from openmind.mcts.constant.mcts_constant import UNIFORM_PRIOR, VALUE_PRIOR
from openmind.mcts.factory.move_prior_factory import create_move_prior
from openmind.parallel.model.dropped_call import DroppedCall
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rbs.service.rule_valuer import RuleValuer
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.training.constant.continuous_constant import CONTINUOUS_GAME
from openmind.training.constant.training_constant import SEED_RANGE
from openmind.training.mapper.played_game_summary_mapper import PlayedGameSummaryMapper
from openmind.training.model.continuous_training_settings import ContinuousTrainingSettings
from openmind.training.model.game_lesson import GameLesson
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.study_snapshot import StudySnapshot
from openmind.training.service.arm_selector import ArmSelector
from openmind.training.service.game_replayer import GameReplayer
from openmind.training.service.game_study import GameStudy
from openmind.training.service.lesson_learner import LessonLearner
from openmind.training.service.rule_searcher import RuleSearcher
from openmind.training.service.self_play import SelfPlay
from openmind.training.service.signal_preparer import SignalPreparer

logger = logging.getLogger(__name__)


class ContinuousTrainer:
    """Trains value rules from games played one after another, learning from each game as it ends, without rounds.

    Games are played between arms, the value bases the signal library holds, in the task runner's workers, each worker
    taking the next game as soon as it has studied the last. When a worker takes a game, UCB picks its two arms from the
    scores the game memory counts, and the game plays with the library, the weights and the seeds as they stand at that
    moment. No search tree outlives its move.

    As each game's lesson arrives here, it is remembered with its arms' models, learned from (records, proofs and seeds,
    one weight step per arm), and the library is handed to `on_game` to be saved. A decisive game is followed by a rule
    search over every game played so far, and no new game starts until the search's rules are in the library; games
    already under way finish meanwhile, and their lessons are learned once the search is done.

    Without a library holding value bases, the signals deduced from the rules come first, as the arms."""

    def __init__(
        self,
        game_study: GameStudy,
        lesson_learner: LessonLearner,
        rule_searcher: RuleSearcher,
        game_memory: GameMemory,
        game_replayer: GameReplayer,
        signal_preparer: SignalPreparer,
        self_play: SelfPlay,
        arm_selector: ArmSelector,
        task_runner: TaskRunner,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
        played_game_summary_mapper: PlayedGameSummaryMapper | None = None,
    ) -> None:
        self._game_study = game_study
        self._lesson_learner = lesson_learner
        self._rule_searcher = rule_searcher
        self._game_memory = game_memory
        self._game_replayer = game_replayer
        self._signal_preparer = signal_preparer
        self._self_play = self_play
        self._arm_selector = arm_selector
        self._task_runner = task_runner
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library
        self._summaries = PlayedGameSummaryMapper() if played_game_summary_mapper is None else played_game_summary_mapper

    def train(
        self,
        domain: Domain,
        library: SignalLibrary | None,
        settings: ContinuousTrainingSettings,
        on_game: Callable[[SignalLibrary], None] | None = None,
    ) -> SignalLibrary:
        """Plays `settings.games` games, or until stopped, and gives the library as it stands after the last."""
        if len(domain.players.names) != 2:
            raise ValueError(f"Continuous training plays games between two players, not {len(domain.players.names)}")
        if library is None or len(library.value_bases) < 2:
            prepared = self._signal_preparer.prepare(domain, settings.signals.goal_limit)
            known = set() if library is None else {record.signal.name for record in library.records}
            records = prepared.records if library is None else (*library.records, *(record for record in prepared.records if record.signal.name not in known))
            library = replace(prepared, records=records, supports=() if library is None else library.supports)
        rng = random.Random(settings.seed)
        state = {"library": library}
        first = self._game_memory.count(CONTINUOUS_GAME)
        pending: dict[str, int] = {}
        seated: dict[int, tuple[tuple[str, str], tuple[ModelDescription, ...]]] = {}
        logger.info(
            "Training continuously from game %d on %s: %d arms, learning rate %s, a rule search of at most %s seconds "
            "after each decisive game",
            first + 1,
            "until stopped" if settings.games is None else f"{settings.games} games",
            len(library.value_bases),
            settings.learning_rate,
            settings.values.seconds,
        )

        def arguments_for(index: int) -> tuple[object, ...]:
            current = state["library"]
            bases = dict(current.value_bases)
            arms = list(bases)
            scores = {name: (games, wins + draws / 2) for name, (games, wins, draws, _) in self._game_memory.scores(arms).items()}
            pair = self._arm_selector.pair(scores, pending, arms, settings.signals.exploration, rng)
            order = pair if index % 2 == 0 else (pair[1], pair[0])
            builders = tuple(self._agent_builder(domain, bases[arm], settings) for arm in order)
            seated[index] = (order, tuple(builder.describe(arm) for builder, arm in zip(builders, order, strict=True)))
            for arm in order:
                pending[arm] = pending.get(arm, 0) + 1
            best = max(arms, key=lambda name: scores[name][1] / scores[name][0] if scores[name][0] else 0.0)
            snapshot = StudySnapshot(current, self._lesson_learner.seeds, bases[best])
            return domain, builders, order, rng.randrange(SEED_RANGE), rng.randrange(SEED_RANGE), snapshot, settings

        def on_result(index: int, lesson: GameLesson | DroppedCall) -> None:
            order, models = seated.pop(index)
            for arm in order:
                pending[arm] -= 1
            if isinstance(lesson, DroppedCall):
                return
            game = lesson.game
            number = first + index + 1
            current = state["library"]
            record = next(iter(self._self_play.records(domain, (game,))), None)
            summary = self._summaries.to_summary(domain, game, CONTINUOUS_GAME, None, number, models, record)
            self._game_memory.remember(summary)
            current = self._lesson_learner.learn(domain, current, lesson, summary.label, settings)
            if len(set(game.payoffs)) > 1:
                logger.info("Game %d was decisive: searching for rules before the next game starts", number)
                games = tuple(self._game_replayer.replay(domain, remembered) for remembered in self._game_memory.games(CONTINUOUS_GAME))
                current = self._rule_searcher.search(
                    domain, current, games, self._lesson_learner.deductions, self._lesson_learner.seeds, settings
                )
            state["library"] = current
            if on_game is not None:
                on_game(current)

        self._task_runner.stream(
            self._game_study.play_and_study,
            settings.games,
            arguments_for,
            on_result,
            droppable=True,
            keep_results=False,
        )
        return state["library"]

    def _agent_builder(self, domain: Domain, value_base: ValueBase, settings: ContinuousTrainingSettings) -> AgentBuilder:
        """An agent following an arm: valuing the positions its rollouts reach with the arm's value base after the
        rollout actions, falling back on the deduction, on the time control when there is one, selecting as the settings
        say."""
        valuer = RuleValuer(value_base, domain, self._rule_compiler, self._rule_runner, self._consequence_library)
        builder = (
            AgentBuilder()
            .with_exploration(EXPLORATION)
            .with_iterations(settings.iterations)
            .with_rollout_limit(settings.rollout_limit, settings.unfinished_payoff)
            .with_deduction(settings.deduction)
            .with_valuation(valuer)
            .with_rollout_actions(settings.rollout_actions)
        )
        if settings.time_control is not None:
            builder.with_time_budget_estimator(PlainTimeBudgetEstimator(settings.expected_steps))
        kind = VALUE_PRIOR if settings.prior == VALUE_PRIOR else UNIFORM_PRIOR
        builder.with_selection(settings.selection, settings.puct_exploration).with_prior(
            create_move_prior(kind, settings.prior_temperature, domain, None, valuer)
        )
        return builder
