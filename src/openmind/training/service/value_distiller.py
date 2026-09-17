import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import ARMS_GAME, HELD_OUT_ARMS_GAME, HELD_OUT_SELF_PLAY_GAME, SELF_PLAY_GAME
from openmind.agent.model.domain import Domain
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.service.game_memory import GameMemory
from openmind.inference.model.expression import Expression
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_generation_result import ValueGenerationResult
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rbs.service.rule_valuer import RuleValuer
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.training.constant.signal_constant import UNIFORM, WEIGHTED, WIN
from openmind.training.constant.training_constant import SEARCH_TARGET, SIGNALS_TARGET
from openmind.training.mapper.played_game_summary_mapper import PlayedGameSummaryMapper
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.pondering import Pondering
from openmind.training.model.pondering_summary import PonderingSummary
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_record import SignalRecord
from openmind.training.model.signal_settings import SignalSettings
from openmind.training.model.value_distillation_result import ValueDistillationResult
from openmind.training.model.value_distillation_settings import ValueDistillationSettings
from openmind.training.service.arm_selector import ArmSelector
from openmind.training.service.candidate_signals import CandidateSignals
from openmind.training.service.position_ponderer import PositionPonderer
from openmind.training.service.self_play import SelfPlay
from openmind.training.service.signal_library_updater import SignalLibraryUpdater
from openmind.training.service.signal_ranker import SignalRanker
from openmind.training.service.signal_recorder import SignalRecorder
from openmind.training.service.signal_targeter import SignalTargeter
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class ValueDistiller:
    """Distills value rules from self-play: turns the positions of training and held-out games into rows valued at the
    target, ponders the training positions the previous rules missed most when the settings say so, fits value rules on
    the training rows, trying the seeds pondering induced first, chooses among the fits on the held-out rows, and
    measures the chosen rules on the held-out rows. With a game memory, every game is remembered as it ends, with the
    model each player played, and arms' scores are counted from what it remembers."""

    def __init__(
        self,
        self_play: SelfPlay,
        value_generator: ValueGenerator,
        position_row_mapper: PositionRowMapper,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
        position_ponderer: PositionPonderer,
        expression_generator: ExpressionGenerator,
        signal_recorder: SignalRecorder,
        signal_ranker: SignalRanker,
        signal_targeter: SignalTargeter,
        signal_library_updater: SignalLibraryUpdater,
        arm_selector: ArmSelector,
        game_memory: GameMemory | None = None,
        played_game_summary_mapper: PlayedGameSummaryMapper | None = None,
    ) -> None:
        self._arm_selector = arm_selector
        self._game_memory = game_memory
        self._summaries = PlayedGameSummaryMapper() if played_game_summary_mapper is None else played_game_summary_mapper
        self._self_play = self_play
        self._value_generator = value_generator
        self._position_row_mapper = position_row_mapper
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library
        self._position_ponderer = position_ponderer
        self._expression_generator = expression_generator
        self._signal_recorder = signal_recorder
        self._signal_ranker = signal_ranker
        self._signal_targeter = signal_targeter
        self._signal_library_updater = signal_library_updater

    def distill(
        self,
        domain: Domain,
        agent_builder: AgentBuilder,
        settings: ValueDistillationSettings,
        value_base: ValueBase | None = None,
        library: SignalLibrary | None = None,
        arm_builders: Mapping[str, AgentBuilder] | None = None,
        round_number: int | None = None,
        model_name: str = SELF_PLAY_GAME,
    ) -> ValueDistillationResult:
        """Sets the builder's iterations; the builder needs its exploration set. Pondering measures misses against the
        given value base, the previous round's rules, or the mean target without one. With the signals target, the
        signals are recorded in the given library, a new one without it, and the result holds the library updated; given
        two arm builders or more, by signal name, self-play games are between arms, each game's pair chosen by UCB when a
        worker starts it, and every game's result goes to its arms' records. `round_number` and `model_name`, what the
        self-play agent is called, name the games a game memory remembers."""
        rng = random.Random(settings.seed)
        agent_builder.with_iterations(settings.iterations)
        clocked = settings.time_control is not None
        if clocked:
            agent_builder.with_time_budget_estimator(PlainTimeBudgetEstimator(settings.expected_steps))
        if settings.target == SIGNALS_TARGET and arm_builders is not None and len(arm_builders) >= 2:
            exploration = (settings.signals or SignalSettings()).exploration
            library = library or SignalLibrary(domain.name)
            for builder in arm_builders.values():
                builder.with_iterations(settings.iterations)
                if clocked:
                    builder.with_time_budget_estimator(PlainTimeBudgetEstimator(settings.expected_steps))
            arms = {name: builder.describe(name) for name, builder in arm_builders.items()}
            training_games = self._self_play.play_arms(
                domain,
                arm_builders,
                settings.games,
                self._scores(library, arm_builders),
                self._arm_selector,
                exploration,
                rng,
                keep_samples=False,
                time_control=settings.time_control,
                on_game=self._remembering(domain, ARMS_GAME, round_number, lambda game: tuple(arms[arm] for arm in game.arms)),
            )
            library = self._signal_library_updater.score(library, training_games)
            held_out_games = self._self_play.play_arms(
                domain,
                arm_builders,
                settings.held_out_games,
                self._scores(library, arm_builders),
                self._arm_selector,
                exploration,
                rng,
                keep_samples=False,
                time_control=settings.time_control,
                on_game=self._remembering(
                    domain, HELD_OUT_ARMS_GAME, round_number, lambda game: tuple(arms[arm] for arm in game.arms)
                ),
            )
            library = self._signal_library_updater.score(library, held_out_games)
            return self._distill_signals(domain, settings, value_base, library, training_games, held_out_games)
        model = agent_builder.describe(model_name)
        players = len(domain.players.names)
        training_games = self._self_play.play(
            domain,
            agent_builder,
            settings.games,
            rng,
            keep_samples=False,
            time_control=settings.time_control,
            on_game=self._remembering(domain, SELF_PLAY_GAME, round_number, lambda game: (model,) * players),
        )
        held_out_games = self._self_play.play(
            domain,
            agent_builder,
            settings.held_out_games,
            rng,
            keep_samples=False,
            time_control=settings.time_control,
            on_game=self._remembering(domain, HELD_OUT_SELF_PLAY_GAME, round_number, lambda game: (model,) * players),
        )
        if settings.target == SIGNALS_TARGET:
            return self._distill_signals(domain, settings, value_base, library, training_games, held_out_games)
        training = self._position_row_mapper.to_rows(domain, training_games, settings.target)
        held_out = self._position_row_mapper.to_rows(domain, held_out_games, settings.target)
        pondering = None
        if settings.pondering is not None:
            pondering = self._position_ponderer.ponder(domain, training, value_base, settings.pondering, training_games)
            training = pondering.rows
        seeds = () if pondering is None else pondering.seeds
        generation = self._value_generator.generate(domain, training, held_out, settings.values, seeds)
        held_out_error = self._error(domain, generation.value_base, held_out)
        logger.info(
            "Distilled %d value rules from %d training rows valued at the %s target; mean absolute error %s on %d "
            "held-out rows",
            len(generation.value_base.rules),
            len(training),
            settings.target,
            held_out_error,
            len(held_out),
        )
        summary = None if pondering is None else self._summary(pondering, generation)
        if summary is not None:
            logger.info(
                "Pondering: %d positions, %d proven; %d seeds, %d kept by the search, %d in the value rules; %d positions "
                "of decisive games deduced walking back from their ends, %d proven",
                summary.positions,
                summary.proven,
                summary.seeds,
                summary.seeds_kept,
                summary.seeds_in_rules,
                summary.endings_deduced,
                summary.endings_proven,
            )
        return ValueDistillationResult(
            generation.value_base,
            generation.fits,
            generation.chosen,
            generation.candidates,
            len(training),
            len(held_out),
            held_out_error,
            summary,
            records=self._self_play.records(domain, training_games),
        )

    def _distill_signals(
        self,
        domain: Domain,
        settings: ValueDistillationSettings,
        value_base: ValueBase | None,
        library: SignalLibrary | None,
        training_games: Sequence[PlayedGame],
        held_out_games: Sequence[PlayedGame],
    ) -> ValueDistillationResult:
        """Records every candidate signal on the round's anchors, follows winning, the signals with the best records and
        the two aggregations, fits a value base per followed signal from one search, updates the rules' support, and
        returns the weighted aggregation's value base as the round's rules."""
        signal_settings = settings.signals or SignalSettings()
        pondering = None
        if settings.pondering is not None:
            search_rows = self._position_row_mapper.to_rows(domain, training_games, SEARCH_TARGET)
            pondering = self._position_ponderer.ponder(domain, search_rows, value_base, settings.pondering, training_games)
        deductions = (
            ()
            if pondering is None
            else (*pondering.deductions, *(deduction for walk in pondering.walks for deduction in walk.deductions))
        )
        seeds = () if pondering is None else pondering.seeds
        library = library or SignalLibrary(domain.name)
        recorded = (Signal(WIN), *self._candidates(domain, training_games, seeds, library))
        readings = self._signal_recorder.read(domain, recorded, training_games, deductions)
        library = self._signal_recorder.add(library, recorded, readings)
        best = self._signal_ranker.best(library, signal_settings.arms)
        followed = (Signal(WIN), *(record.signal for record in best))
        reliabilities = self._signal_ranker.reliabilities(library, followed)
        parts = tuple(signal.name for signal in followed)
        aggregations = (Signal(UNIFORM, parts=parts), Signal(WEIGHTED, parts=parts))
        readings = self._signal_recorder.aggregate(readings, aggregations[0])
        readings = self._signal_recorder.aggregate(readings, aggregations[1], reliabilities)
        library = self._signal_recorder.add(library, aggregations, readings)
        signals = (*followed, *aggregations)
        records = {record.signal.name: record for record in library.records}
        arms = tuple(records[signal.name] for signal in signals)
        for record in arms:
            logger.info(
                "Following %s: reliability %s from %d agreements and %d disagreements",
                record.signal.name,
                record.reliability,
                record.agreements,
                record.disagreements,
            )

        training = self._signal_targeter.rows(domain, training_games)
        held_out = self._signal_targeter.rows(domain, held_out_games)
        training_targets = self._signal_targeter.targets(
            domain, training_games, signals, reliabilities, signal_settings.horizon, deductions
        )
        held_out_targets = self._signal_targeter.targets(domain, held_out_games, signals, reliabilities, signal_settings.horizon)
        targets = {
            name: (values, held_out_targets[name]) for name, values in training_targets.items() if name in held_out_targets
        }
        generations = self._value_generator.generate_for_targets(domain, training, held_out, targets, settings.values, seeds)
        library = self._signal_library_updater.update(library, generations)
        chosen = self._round_signal(library, generations)
        generation = generations[chosen]
        chosen_held_out = tuple(
            replace(row, target=float(value)) for row, value in zip(held_out, held_out_targets[chosen], strict=True)
        )
        held_out_error = self._error(domain, generation.value_base, chosen_held_out)
        logger.info(
            "Distilled %d value rules from %d training rows valued at the %s target, the %s signal's; mean absolute error "
            "%s on %d held-out rows",
            len(generation.value_base.rules),
            len(training),
            settings.target,
            chosen,
            held_out_error,
            len(held_out),
        )
        summary = None if pondering is None else self._summary(pondering, generation)
        return ValueDistillationResult(
            generation.value_base,
            generation.fits,
            generation.chosen,
            generation.candidates,
            len(training),
            len(held_out),
            held_out_error,
            summary,
            library,
            arms,
            self._self_play.records(domain, training_games),
        )

    def _round_signal(self, library: SignalLibrary, generations: Mapping[str, ValueGenerationResult]) -> str:
        """The signal whose rules are the round's: the fitted signal with the best game score, ties going to the one that
        played most; the weighted aggregation while no fitted signal has played a game."""
        records = {record.signal.name: record for record in library.records}
        played = [name for name in generations if name in records and records[name].games > 0]
        if not played:
            logger.info("The round's rules are the %s signal's: no signal followed has played a game yet", WEIGHTED)
            return WEIGHTED
        best = max(played, key=lambda name: (self._score(records[name]), records[name].games))
        logger.info(
            "The round's rules are the %s signal's: score %s over %d games, the best of the signals followed",
            best,
            self._score(records[best]),
            records[best].games,
        )
        return best

    def _scores(self, library: SignalLibrary, arms: Mapping[str, AgentBuilder]) -> dict[str, tuple[int, float]]:
        """Every arm's games and points so far, by signal name: counted from the game memory's outcomes when there is
        one, from the library's records otherwise."""
        if self._game_memory is not None:
            return {
                name: (games, wins + draws / 2) for name, (games, wins, draws, _) in self._game_memory.scores(arms).items()
            }
        return {record.signal.name: (record.games, record.wins + record.draws / 2) for record in library.records}

    def _remembering(
        self,
        domain: Domain,
        kind: str,
        round_number: int | None,
        models: Callable[[PlayedGame], tuple[ModelDescription, ...]],
    ) -> Callable[[int, PlayedGame], None] | None:
        """What remembers each game as it ends, the models each player played given by `models`; None without a game
        memory."""
        memory = self._game_memory
        if memory is None:
            return None

        def remember(index: int, game: PlayedGame) -> None:
            record = next(iter(self._self_play.records(domain, (game,))), None)
            memory.remember(self._summaries.to_summary(domain, game, kind, round_number, index + 1, models(game), record))

        return remember

    def _score(self, record: SignalRecord) -> float:
        return (record.wins + record.draws / 2) / record.games

    def _candidates(
        self, domain: Domain, games: Sequence[PlayedGame], seeds: Sequence[Expression], library: SignalLibrary
    ) -> tuple[Signal, ...]:
        return CandidateSignals(self._expression_generator).candidates(domain, games, seeds, library)

    def _summary(self, pondering: Pondering, generation: ValueGenerationResult) -> PonderingSummary:
        sources = set(pondering.sources)
        return PonderingSummary(
            len(pondering.deductions),
            sum(1 for deduction in pondering.deductions if deduction.proven),
            len(pondering.seeds),
            sum(1 for candidate in generation.candidates if candidate in sources),
            sum(1 for rule in generation.value_base.rules if rule.term in sources),
            sum(len(walk.deductions) for walk in pondering.walks),
            sum(walk.proven for walk in pondering.walks),
        )

    def _error(self, domain: Domain, value_base: ValueBase, rows: Sequence[PositionRow]) -> float | None:
        valuer = RuleValuer(value_base, domain, self._rule_compiler, self._rule_runner, self._consequence_library)
        values_by_state: dict[State, tuple[float, ...] | None] = {}
        errors: list[float] = []
        for row in rows:
            if row.state not in values_by_state:
                values_by_state[row.state] = valuer.value(row.state)
            values = values_by_state[row.state]
            if values is not None:
                errors.append(abs(values[domain.players.names.index(row.player)] - row.target))
        return math.fsum(errors) / len(errors) if errors else None
