import logging
import random
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from functools import partial

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.model.domain import Domain
from openmind.agent.model.policy_factory import PolicyFactory
from openmind.evaluation.constant.evaluation_constant import RANDOM_OPPONENT, UNTRAINED_OPPONENT
from openmind.evaluation.factory.baseline_policy_factory import create_random_policy, create_seeded_agent
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rbs.service.rule_valuer import RuleValuer
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.training.constant.training_constant import SIGNALS_TARGET, START_RULES
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_settings import SignalSettings
from openmind.training.model.training_report import TrainingReport
from openmind.training.model.training_round import TrainingRound
from openmind.training.model.value_training_settings import ValueTrainingSettings
from openmind.training.service.signal_preparer import SignalPreparer
from openmind.training.service.value_distiller import ValueDistiller

logger = logging.getLogger(__name__)


class ValueTrainingLoop:
    """Trains value rules round after round from self-play. In each round, self-play agents value the positions their
    rollouts reach with the previous round's rules, or the start rules, after a few rollout actions, and deduce the
    positions those rules have no clue about when the settings give a deduction budget; the round ponders the positions
    the previous rules missed most, when the distillation says so, and fits new rules; then the new rules' agent plays
    the random policy, untrained MCTS and the previous round's agent, every agent searching with its game's own seed.
    Round k seeds with the distillation's seed plus k."""

    def __init__(
        self,
        value_distiller: ValueDistiller,
        match_runner: MatchRunner,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
        signal_preparer: SignalPreparer,
    ) -> None:
        self._value_distiller = value_distiller
        self._match_runner = match_runner
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library
        self._signal_preparer = signal_preparer

    def train(
        self,
        domain: Domain,
        start: ValueBase | None,
        settings: ValueTrainingSettings,
        on_round: Callable[[TrainingReport], None] | None = None,
        library: SignalLibrary | None = None,
    ) -> TrainingReport:
        """Hands the report so far to on_round once a round's rules are fitted, before its games, so they can be saved
        even if the games never end, and again after every round; the last report is complete. To run in several
        workers, the value rules' agents must pickle, as a RuleValuer does. With the signals target, round 1 records its
        signals in the given library, a new one without it, and every round hands the library it updated to the next;
        each round holds it. Once the library holds two value bases or more, the signals the previous round followed,
        self-play games are between agents each following one of them. With the signals target and a library without value
        bases, the signal preparer first deduces signals from the rules: their records join the library's, and round 1's
        arms follow the deduced value bases."""
        created_at = datetime.now().replace(microsecond=0)
        if settings.distillation.target == SIGNALS_TARGET and (library is None or not library.value_bases):
            prepared = self._signal_preparer.prepare(domain, (settings.distillation.signals or SignalSettings()).goal_limit)
            if library is None:
                library = prepared
            else:
                known = {record.signal.name for record in library.records}
                records = (*library.records, *(record for record in prepared.records if record.signal.name not in known))
                library = replace(library, records=records, value_bases=prepared.value_bases)
        previous, previous_name = start, None if start is None else START_RULES
        rounds: list[TrainingRound] = []
        report = TrainingReport(domain.name, created_at, settings, (), settings.rounds == 0)
        for number in range(1, settings.rounds + 1):
            started = time.perf_counter()
            if previous is None:
                logger.info("Round %d of %d: self-play without value rules", number, settings.rounds)
            else:
                logger.info(
                    "Round %d of %d: self-play valuing positions with %s",
                    number,
                    settings.rounds,
                    f"the {previous_name}" if previous_name == START_RULES else f"{previous_name}'s rules",
                )
            seed = settings.distillation.seed + number
            arm_builders = (
                None
                if library is None or len(library.value_bases) < 2
                else {name: self._agent_builder(domain, base, settings) for name, base in library.value_bases}
            )
            result = self._value_distiller.distill(
                domain,
                self._agent_builder(domain, previous, settings),
                replace(settings.distillation, seed=seed),
                previous,
                library,
                arm_builders,
            )
            library = result.library if result.library is not None else library
            logger.info(
                "Round %d: %d value rules, held-out loss %s, held-out error %s",
                number,
                len(result.value_base.rules),
                None if result.chosen is None else result.chosen.held_out_loss,
                result.held_out_error,
            )
            baselines: tuple[MatchResults, ...] = ()
            against_previous: MatchResults | None = None
            if settings.evaluation_games and on_round is not None:
                fitted = TrainingRound(
                    number,
                    result.value_base,
                    result.fits,
                    result.chosen,
                    result.training_rows,
                    result.held_out_rows,
                    result.held_out_error,
                    (),
                    None,
                    time.perf_counter() - started,
                    result.pondering,
                    result.arms,
                    result.library,
                )
                logger.info("Round %d fitted: handing it over before its games", number)
                on_round(TrainingReport(domain.name, created_at, settings, (*rounds, fitted), False))
            if settings.evaluation_games:
                rng = random.Random(seed)
                evaluated = partial(create_seeded_agent, self._agent_builder(domain, result.value_base, settings))
                opponents: tuple[tuple[str, PolicyFactory], ...] = (
                    (RANDOM_OPPONENT, create_random_policy),
                    (UNTRAINED_OPPONENT, partial(create_seeded_agent, self._agent_builder(domain, None, settings, False))),
                )
                baselines = tuple(
                    self._series(domain, number, evaluated, name, opponent, settings.evaluation_games, rng)
                    for name, opponent in opponents
                )
                if previous is not None and previous_name is not None:
                    previous_agent = partial(create_seeded_agent, self._agent_builder(domain, previous, settings))
                    against_previous = self._series(
                        domain, number, evaluated, previous_name, previous_agent, settings.evaluation_games, rng
                    )
            seconds = time.perf_counter() - started
            logger.info("Round %d took %.0f seconds", number, seconds)
            rounds.append(
                TrainingRound(
                    number,
                    result.value_base,
                    result.fits,
                    result.chosen,
                    result.training_rows,
                    result.held_out_rows,
                    result.held_out_error,
                    baselines,
                    against_previous,
                    seconds,
                    result.pondering,
                    result.arms,
                    result.library,
                )
            )
            previous, previous_name = result.value_base, f"round {number}"
            report = TrainingReport(domain.name, created_at, settings, tuple(rounds), number == settings.rounds)
            if on_round is not None:
                on_round(report)
        return report

    def _agent_builder(
        self, domain: Domain, value_base: ValueBase | None, settings: ValueTrainingSettings, deduces: bool = True
    ) -> AgentBuilder:
        """An agent searching with the settings' iterations and rollout limit; with a value base, valuing the positions
        its rollouts reach after the settings' rollout actions; and, unless told not to, falling back on the settings'
        deduction when its rules have no clue. Untrained MCTS never deduces."""
        builder = AgentBuilder().with_exploration(EXPLORATION).with_iterations(settings.distillation.iterations)
        builder.with_rollout_limit(settings.rollout_limit, settings.unfinished_payoff)
        builder.with_deduction(settings.deduction if deduces else None)
        if value_base is not None:
            valuer = RuleValuer(value_base, domain, self._rule_compiler, self._rule_runner, self._consequence_library)
            builder.with_valuation(valuer).with_rollout_actions(settings.rollout_actions)
        return builder

    def _series(
        self,
        domain: Domain,
        number: int,
        evaluated: PolicyFactory,
        name: str,
        opponent: PolicyFactory,
        games: int,
        rng: random.Random,
    ) -> MatchResults:
        results = self._match_runner.series(domain, evaluated, opponent, name, games, rng)
        logger.info(
            "Round %d against %s: %d games, %d wins, %d draws, %d losses",
            number,
            name,
            results.games,
            results.wins,
            results.draws,
            results.losses,
        )
        return results
