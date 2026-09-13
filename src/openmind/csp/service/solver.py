import logging
from collections import OrderedDict
from operator import itemgetter

from openmind.csp.constant.solver_constant import CACHE_SIZE
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.csp.model.support_table import SupportTable
from openmind.csp.model.variable import Variable
from openmind.csp.service.backtracking_search import BacktrackingSearch
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.mapper.parameter_scope_mapper import ParameterScopeMapper
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_different import AllDifferent
from openmind.expression.model.expression import Expression
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Solver:
    """Finds the actions whose parameter values satisfy all of their constraints in a state. Each constraint takes the
    strongest form its scope allows (a single check, a domain filter, an all-different group, a support table, or a
    forward-checked constraint), backtracking search does the rest, and results are cached per problem and state."""

    def __init__(
        self,
        constraint_checker: ConstraintChecker,
        parameter_scope_mapper: ParameterScopeMapper,
        backtracking_search: BacktrackingSearch,
        expression_text_mapper: ExpressionTextMapper,
    ) -> None:
        self._constraint_checker = constraint_checker
        self._parameter_scope_mapper = parameter_scope_mapper
        self._backtracking_search = backtracking_search
        self._expression_text_mapper = expression_text_mapper
        self._cache: OrderedDict[
            tuple[int, State, int | None], tuple[Problem, tuple[tuple[Action, ...], SolveStatistics]]
        ] = OrderedDict()

    def solve(self, problem: Problem, state: State, limit: int | None = None) -> tuple[Action, ...]:
        """Every solution, or at most limit, ordered by the variables' domains with the first variable changing slowest."""
        return self.solve_with_statistics(problem, state, limit)[0]

    def solve_with_statistics(
        self, problem: Problem, state: State, limit: int | None = None
    ) -> tuple[tuple[Action, ...], SolveStatistics]:
        """The solutions solve gives, with what the search did summed over the problem's actions. A cached result keeps
        the statistics of the search that found it."""
        key = (id(problem), state, limit)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is problem:
            self._cache.move_to_end(key)
            return cached[1]
        actions: list[Action] = []
        assignments = dead_ends = pruned_values = 0
        for definition in problem.actions:
            remaining = None if limit is None else limit - len(actions)
            if remaining == 0:
                break
            found, statistics = self._solve_action(definition, state, remaining)
            actions.extend(found)
            assignments += statistics.assignments
            dead_ends += statistics.dead_ends
            pruned_values += statistics.pruned_values
        result = (tuple(actions), SolveStatistics(len(actions), assignments, dead_ends, pruned_values))
        self._cache[key] = (problem, result)
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return result

    def _solve_action(
        self, definition: ActionDefinition, state: State, limit: int | None
    ) -> tuple[list[Action], SolveStatistics]:
        action = definition.name
        names = [variable.name for variable in definition.variables]
        domains = {variable.name: variable.domain.values for variable in definition.variables}
        scoped = [(constraint, self._parameter_scope_mapper.to_scope(constraint)) for constraint in definition.constraints]
        for _, scope in scoped:
            unknown = sorted(scope.difference(names))
            if unknown:
                raise KeyError(f"Unknown parameter of {action!r}: {unknown[0]!r}")

        groups: list[AllDifferentGroup] = []
        wider: list[tuple[Expression, frozenset[str]]] = []
        for constraint, scope in scoped:
            if not scope:
                if not self._constraint_checker.holds(constraint, state, action, {}):
                    return self._no_solution(action, constraint)
            elif self._is_group(constraint):
                assert isinstance(constraint, AllDifferent)
                parameters = [operand.name for operand in constraint.operands if isinstance(operand, ActionParameter)]
                fixed = [
                    self._constraint_checker.value(operand, state, action, {})
                    for operand in constraint.operands
                    if not isinstance(operand, ActionParameter)
                ]
                if len(set(parameters)) < len(parameters) or len(set(fixed)) < len(fixed):
                    return self._no_solution(action, constraint)
                for name in parameters:
                    domains[name] = tuple(value for value in domains[name] if value not in fixed)
                if len(parameters) > 1:
                    groups.append(AllDifferentGroup(tuple(parameters)))
            elif len(scope) == 1:
                (name,) = scope
                domains[name] = tuple(
                    value
                    for value in domains[name]
                    if self._constraint_checker.holds(constraint, state, action, {name: value})
                )
            else:
                wider.append((constraint, scope))

        tables: list[SupportTable] = []
        constraints: list[ScopedConstraint] = []
        for constraint, scope in wider:
            ordered_scope = tuple(sorted(scope, key=names.index))
            if len(ordered_scope) == 2:
                first, second = ordered_scope
                allowed = frozenset(
                    (first_value, second_value)
                    for first_value in domains[first]
                    for second_value in domains[second]
                    if self._constraint_checker.holds(
                        constraint, state, action, {first: first_value, second: second_value}
                    )
                )
                tables.append(SupportTable(first, second, allowed))
            else:
                constraints.append(ScopedConstraint(constraint, ordered_scope))

        space = SearchSpace(
            action,
            tuple(Variable(name, DiscreteDomain(domains[name])) for name in names),
            tuple(tables),
            tuple(groups),
            tuple(constraints),
        )
        solutions, statistics = self._backtracking_search.search(space, state, limit)
        position = {
            variable.name: {value: index for index, value in enumerate(variable.domain.values)}
            for variable in definition.variables
        }
        solutions = tuple(
            sorted(solutions, key=lambda solution: tuple(position[name][solution[name]] for name in names))
        )
        logger.debug(
            "%s: %d solutions, %d assignments, %d dead ends, %d values pruned",
            action,
            statistics.solutions,
            statistics.assignments,
            statistics.dead_ends,
            statistics.pruned_values,
        )
        return [Action(action, tuple(sorted(solution.items(), key=itemgetter(0)))) for solution in solutions], statistics

    def _is_group(self, constraint: Expression) -> bool:
        return isinstance(constraint, AllDifferent) and all(
            isinstance(operand, ActionParameter) or not self._parameter_scope_mapper.to_scope(operand)
            for operand in constraint.operands
        )

    def _no_solution(self, action: str, constraint: Expression) -> tuple[list[Action], SolveStatistics]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s: no solution, constraint is false: %s", action, self._expression_text_mapper.to_text(constraint)
            )
        return [], SolveStatistics(0, 0, 0, 0)
