import json
import logging
import random

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.service.game_memory import GameMemory
from openmind.knowledge.constant.knowledge_constant import INFERENCE
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.source import Source
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.mcts.constant.mcts_constant import UNIFORM_PRIOR, VALUE_PRIOR
from openmind.mcts.factory.move_prior_factory import create_move_prior
from openmind.parallel.model.dropped_call import DroppedCall
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.training.constant.continuous_constant import CONTINUOUS_GAME, NO_VALUE_RULES, PROOF_KEYWORD
from openmind.training.constant.training_constant import SEED_RANGE
from openmind.training.mapper.played_game_summary_mapper import PlayedGameSummaryMapper
from openmind.training.model.arm_library import ArmLibrary
from openmind.training.model.continuous_training_settings import ContinuousTrainingSettings
from openmind.training.model.game_lesson import GameLesson
from openmind.training.service.arm_selector import ArmSelector
from openmind.training.service.game_study import GameStudy
from openmind.training.service.self_play import SelfPlay

logger = logging.getLogger(__name__)


class ContinuousTrainer:
    """Plays games one after another between arms, the value bases an arm library holds, in the task runner's workers,
    each worker taking the next game as soon as it has studied the last. When a worker takes a game, UCB picks its two
    arms from the scores the game memory counts. No search tree outlives its move.

    As each game's lesson arrives here, the game is remembered with its arms' models, and every position its walk back
    proved is remembered as `proved`, with the game and the ply of its position. Nothing is learned from the games yet.

    While the library holds fewer than two value bases, the seats they leave are taken by an agent without value rules,
    the arm `no value rules`: its search values nothing, falling back on the deduction where there is one."""

    def __init__(
        self,
        game_study: GameStudy,
        game_memory: GameMemory,
        knowledge_base: KnowledgeBase,
        self_play: SelfPlay,
        arm_selector: ArmSelector,
        task_runner: TaskRunner,
        played_game_summary_mapper: PlayedGameSummaryMapper | None = None,
    ) -> None:
        self._game_study = game_study
        self._game_memory = game_memory
        self._knowledge_base = knowledge_base
        self._self_play = self_play
        self._arm_selector = arm_selector
        self._task_runner = task_runner
        self._summaries = PlayedGameSummaryMapper() if played_game_summary_mapper is None else played_game_summary_mapper

    def train(self, rbs: RuleBasedSystem, library: ArmLibrary | None, settings: ContinuousTrainingSettings) -> None:
        """Plays `settings.games` games, or until stopped."""
        if len(rbs.players().names) != 2:
            raise ValueError(f"Continuous training plays games between two players, not {len(rbs.players().names)}")
        arm_contexts = dict(() if library is None else library.contexts)
        rng = random.Random(settings.seed)
        first = self._game_memory.last_number(CONTINUOUS_GAME)
        pending: dict[str, int] = {}
        seated: dict[int, tuple[tuple[str, str], tuple[ModelDescription, ...]]] = {}
        logger.info(
            "Playing continuously from game %d on %s: %d arms",
            first + 1,
            "until stopped" if settings.games is None else f"{settings.games} games",
            len(arm_contexts),
        )

        def arguments_for(index: int) -> tuple[object, ...]:
            arms = list(arm_contexts)
            if len(arms) >= 2:
                scores = {
                    name: (games, wins + draws / 2) for name, (games, wins, draws, _) in self._game_memory.scores(arms).items()
                }
                pair = self._arm_selector.pair(scores, pending, arms, settings.arm_exploration, rng)
            else:
                pair = (arms[0] if arms else NO_VALUE_RULES, NO_VALUE_RULES)
            order = pair if index % 2 == 0 else (pair[1], pair[0])
            builders = tuple(self._agent_builder(rbs, arm_contexts.get(arm), settings) for arm in order)
            seated[index] = (order, tuple(builder.describe(arm) for builder, arm in zip(builders, order, strict=True)))
            for arm in order:
                pending[arm] = pending.get(arm, 0) + 1
            return rbs, builders, order, rng.randrange(SEED_RANGE), rng.randrange(SEED_RANGE), settings

        def on_result(index: int, lesson: GameLesson | DroppedCall) -> None:
            order, models = seated.pop(index)
            for arm in order:
                pending[arm] -= 1
            if isinstance(lesson, DroppedCall):
                return
            game = lesson.game
            record = next(iter(self._self_play.records(rbs, (game,))), None)
            summary = self._summaries.to_summary(rbs, game, CONTINUOUS_GAME, None, first + index + 1, models, record)
            game = self._game_memory.remember(summary)
            self._remember_proofs(rbs, lesson, summary.label, game.id)

        self._task_runner.stream(
            self._game_study.play_and_study,
            settings.games,
            arguments_for,
            on_result,
            droppable=True,
            keep_results=False,
        )

    def _remember_proofs(self, rbs: RuleBasedSystem, lesson: GameLesson, label: str, game_id: str) -> None:
        context = self._knowledge_base.ensure_context(rbs.context).id
        deduction_mechanism = self._knowledge_base.ensure_mechanism(INFERENCE).id
        plies = {state: ply for ply, state in enumerate(lesson.game.states)}
        proofs = [deduction for deduction in lesson.walk if deduction.payoffs is not None]
        for deduction in proofs:
            ply = plies.get(deduction.state)
            payoffs = json.dumps(list(deduction.payoffs))  # type: ignore[arg-type]
            proof = Source(
                deduction_mechanism,
                (("method", "deduction"), ("game", label), ("ply", ply), ("player", deduction.player)),
                rests_on=(game_id,),
            )
            self._knowledge_base.believe(
                Belief(
                    f"payoffs proved at ply {ply} of {label}",
                    context,
                    payoffs,
                    evidence=(Evidence(payoffs, 1.0, proof),),
                    tags=(("keyword", PROOF_KEYWORD), ("game", label), ("ply", ply)),
                )
            )
        if lesson.walk:
            logger.info("Remembered %s: %d of %d positions walked back proven", label, len(proofs), len(lesson.walk))

    def _agent_builder(
        self, rbs: RuleBasedSystem, context: str | None, settings: ContinuousTrainingSettings
    ) -> AgentBuilder:
        """An agent following an arm: valuing the positions its rollouts reach with the position rules of the arm's own
        context after the rollout actions, none without a context, falling back on the deduction, on the time control
        when there is one, selecting as the settings say; the value prior needs a context."""
        valuer = None if context is None else create_rule_based_system(self._knowledge_base, context)
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
            builder.with_time_budget_estimator(
                PlainTimeBudgetEstimator(settings.expected_steps, settings.time_control.base_seconds * settings.time_reserve)
            )
        kind = VALUE_PRIOR if settings.prior == VALUE_PRIOR and valuer is not None else UNIFORM_PRIOR
        builder.with_selection(settings.selection, settings.puct_exploration).with_prior(
            create_move_prior(kind, settings.prior_temperature, rbs, None, valuer)
        )
        return builder
