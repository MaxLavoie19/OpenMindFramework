import logging
import math
from collections.abc import Sequence

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.inference.model.expression import Expression
from openmind.inference.service.deduction_inducer import DeductionInducer
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.parallel.model.dropped_call import DroppedCall
from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rbs.service.rule_valuer import RuleValuer
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.training.model.pondering import Pondering
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class PositionPonderer:
    """Reasons about the positions the rules know least about:

    1. every training row's miss is how far its target is from the previous rules' value for its player, or from the
       mean target without rules or where the rules can't value the position;
    2. the positions in play with the largest misses are pondered, as many as the settings say, each deduced within the
       budget in the task runner's workers;
    3. a proven position's rows take the proven payoffs as targets;
    4. every proven deduction induces seeds; a seed induced by more positions comes first, since it's likelier to help."""

    def __init__(
        self,
        position_deducer: PositionDeducer,
        deduction_inducer: DeductionInducer,
        expression_generator: ExpressionGenerator,
        solver: Solver,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
        task_runner: TaskRunner,
    ) -> None:
        self._position_deducer = position_deducer
        self._deduction_inducer = deduction_inducer
        self._expression_generator = expression_generator
        self._solver = solver
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library
        self._task_runner = task_runner

    def ponder(
        self, domain: Domain, rows: Sequence[PositionRow], value_base: ValueBase | None, settings: PonderingSettings
    ) -> Pondering:
        if settings.positions <= 0 or not rows:
            return Pondering(tuple(rows), (), (), ())
        misses = self._misses(domain, rows, value_base)
        worst: dict[State, float] = {}
        for row, miss in zip(rows, misses, strict=True):
            worst[row.state] = max(miss, worst.get(row.state, 0.0))
        playable = [state for state in sorted(worst, key=lambda state: -worst[state]) if self._solver.solve(domain.problem, state)]
        states = playable[: settings.positions]
        deductions = tuple(
            deduction
            for deduction in self._task_runner.map(
                self._position_deducer.deduce, [domain] * len(states), states, [settings.budget] * len(states), droppable=True
            )
            if not isinstance(deduction, DroppedCall)
        )
        proven = {deduction.state: deduction for deduction in deductions if deduction.payoffs is not None}
        names = domain.players.names
        pondered = tuple(
            PositionRow(row.state, row.player, proven[row.state].payoffs[names.index(row.player)])  # type: ignore[index]
            if row.state in proven
            else row
            for row in rows
        )
        vocabulary = self._expression_generator.vocabulary(domain, (row.state for row in rows))
        counts: dict[str, tuple[Expression, int]] = {}
        for deduction in proven.values():
            for seed in self._deduction_inducer.seeds(domain, deduction, vocabulary, settings.budget.highest):
                expression, count = counts.get(seed.template, (seed, 0))
                counts[seed.template] = (expression, count + 1)
        seeds = tuple(expression for expression, _ in sorted(counts.values(), key=lambda item: -item[1]))
        logger.info(
            "Pondered %d positions the rules missed most, the largest miss %s: %d proven within %d plies, %d seeds",
            len(deductions),
            worst[states[0]] if states else None,
            len(proven),
            settings.budget.plies,
            len(seeds),
        )
        for seed in seeds:
            logger.debug("Seed %s", self._expression_generator.source(seed).source)
        return Pondering(pondered, deductions, seeds, tuple(self._expression_generator.source(seed) for seed in seeds))

    def _misses(self, domain: Domain, rows: Sequence[PositionRow], value_base: ValueBase | None) -> list[float]:
        mean = math.fsum(row.target for row in rows) / len(rows)
        if value_base is None:
            return [abs(row.target - mean) for row in rows]
        valuer = RuleValuer(value_base, domain, self._rule_compiler, self._rule_runner, self._consequence_library)
        values: dict[State, tuple[float, ...] | None] = {}
        misses: list[float] = []
        for row in rows:
            if row.state not in values:
                values[row.state] = valuer.value(row.state)
            valued = values[row.state]
            value = mean if valued is None else valued[domain.players.names.index(row.player)]
            misses.append(abs(row.target - value))
        return misses
