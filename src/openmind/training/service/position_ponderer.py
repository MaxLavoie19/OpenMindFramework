import logging
import math
from collections.abc import Sequence

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.inference.model.deduction import Deduction
from openmind.inference.model.deduction_budget import DeductionBudget
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
from openmind.training.model.ending_walk import EndingWalk
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.pondering import Pondering
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class PositionPonderer:
    """Reasons about the positions the rules know least about:

    1. with endings, every decisive training game, one whose players' payoffs differ, is walked back from its end in the
       task runner's workers: its positions are deduced from the last one backward until one isn't proven, or its share
       of the endings is used, the endings divided evenly between the decisive games, at least 1 each;
    2. every training row's miss is how far its target is from the previous rules' value for its player, or from the
       mean target without rules or where the rules can't value the position;
    3. the positions in play with the largest misses, leaving out those the walks proved, are pondered, as many as the
       settings say, each deduced within the budget in the task runner's workers;
    4. a proven position's rows take the proven payoffs as targets;
    5. every proven deduction induces seeds; a seed induced by more positions comes first, since it's likelier to help."""

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
        self,
        domain: Domain,
        rows: Sequence[PositionRow],
        value_base: ValueBase | None,
        settings: PonderingSettings,
        games: Sequence[PlayedGame] = (),
    ) -> Pondering:
        """`games` are the training games the rows come from, walked back with endings."""
        if not rows or (settings.positions <= 0 and (settings.endings <= 0 or not games)):
            return Pondering(tuple(rows), (), (), ())
        walks = self._walks(domain, games, settings)
        walked = {
            deduction.state: deduction for walk in walks for deduction in walk.deductions if deduction.payoffs is not None
        }
        deductions: tuple[Deduction, ...] = ()
        if settings.positions > 0:
            misses = self._misses(domain, rows, value_base)
            worst: dict[State, float] = {}
            for row, miss in zip(rows, misses, strict=True):
                worst[row.state] = max(miss, worst.get(row.state, 0.0))
            playable = [
                state
                for state in sorted(worst, key=lambda state: -worst[state])
                if state not in walked and self._solver.solve(domain.problem, state)
            ]
            states = playable[: settings.positions]
            deductions = tuple(
                deduction
                for deduction in self._task_runner.map(
                    self._position_deducer.deduce, [domain] * len(states), states, [settings.budget] * len(states), droppable=True
                )
                if not isinstance(deduction, DroppedCall)
            )
            logger.info(
                "Pondered %d positions the rules missed most, the largest miss %s: %d proven within %d plies",
                len(deductions),
                worst[states[0]] if states else None,
                sum(1 for deduction in deductions if deduction.payoffs is not None),
                settings.budget.plies,
            )
        proven = walked | {deduction.state: deduction for deduction in deductions if deduction.payoffs is not None}
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
        logger.info("Proofs induced %d seeds from %d proven positions", len(seeds), len(proven))
        for seed in seeds:
            logger.debug("Seed %s", self._expression_generator.source(seed).source)
        return Pondering(
            pondered,
            deductions,
            seeds,
            tuple(self._expression_generator.source(seed) for seed in seeds),
            tuple(walks),
        )

    def walk_back(
        self, domain: Domain, states: Sequence[State], budget: DeductionBudget, limit: int
    ) -> tuple[Deduction, ...]:
        """One game's positions deduced from the last one backward, stopping after the first one not proven or at the
        limit."""
        deductions: list[Deduction] = []
        for state in reversed(states):
            if len(deductions) >= limit:
                break
            deduction = self._position_deducer.deduce(domain, state, budget)
            deductions.append(deduction)
            if deduction.payoffs is None:
                break
        return tuple(deductions)

    def _walks(self, domain: Domain, games: Sequence[PlayedGame], settings: PonderingSettings) -> list[EndingWalk]:
        if settings.endings <= 0:
            return []
        decisive = [(index, game) for index, game in enumerate(games) if len(set(game.payoffs)) > 1]
        if not decisive:
            logger.info("No decisive game among %d to walk back from", len(games))
            return []
        count, limit = len(decisive), max(1, settings.endings // len(decisive))
        results = self._task_runner.map(
            self.walk_back,
            [domain] * count,
            [game.states for _, game in decisive],
            [settings.budget] * count,
            [limit] * count,
            droppable=True,
        )
        walks = [
            EndingWalk(index, result)
            for (index, _), result in zip(decisive, results, strict=True)
            if not isinstance(result, DroppedCall)
        ]
        logger.info(
            "Walked back %d decisive games of %d, %d positions each at most: %d of %d positions proven, the longest proven "
            "line %d positions",
            len(walks),
            len(games),
            limit,
            sum(walk.proven for walk in walks),
            sum(len(walk.deductions) for walk in walks),
            max((walk.proven for walk in walks), default=0),
        )
        return walks

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
