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
from openmind.rule.constant.rule_constant import ALL_DIFFERENT
from openmind.rule.mapper.call_operand_mapper import CallOperandMapper
from openmind.rule.model.compiled_rule import CompiledRule
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Solver:
    """Finds the actions whose parameter values satisfy all of their constraints in a state. Each constraint, a Python
    rule, takes the strongest form the parameters it reads allow (a single check, a domain filter, an all-different
    group, a support table, or a forward-checked constraint), backtracking search does the rest, and results are cached
    per problem and state."""

    def __init__(
        self,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        call_operand_mapper: CallOperandMapper,
        constraint_checker: ConstraintChecker,
        backtracking_search: BacktrackingSearch,
    ) -> None:
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._call_operand_mapper = call_operand_mapper
        self._constraint_checker = constraint_checker
        self._backtracking_search = backtracking_search
        self._cache: OrderedDict[
            tuple[int, State, int | None], tuple[Problem, tuple[tuple[Action, ...], SolveStatistics]]
        ] = OrderedDict()

    def __getstate__(self) -> dict[str, object]:
        """The cache stays behind when the solver is copied to another process: its keys are this process's ids."""
        return {name: value for name, value in self.__dict__.items() if name != "_cache"}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._cache = OrderedDict()

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
            found, statistics = self._solve_action(definition, state, remaining, problem.definitions)
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
        self, definition: ActionDefinition, state: State, limit: int | None, definitions: PythonRule | None
    ) -> tuple[list[Action], SolveStatistics]:
        action = definition.name
        names = [variable.name for variable in definition.variables]
        domains = {variable.name: variable.domain.values for variable in definition.variables}

        groups: list[AllDifferentGroup] = []
        wider: list[CompiledRule] = []
        for constraint in definition.constraints:
            compiled = self._rule_compiler.compile_value(constraint, names, definitions)
            scope = compiled.arguments
            group = self._group(constraint, names, definitions) if scope else None
            if not scope:
                if not self._constraint_checker.holds(compiled, state, action, {}):
                    return self._no_solution(action, constraint)
            elif group is not None:
                parameters, fixed_rules = group
                fixed = [self._rule_runner.value(rule, state) for rule in fixed_rules]
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
                    if self._constraint_checker.holds(compiled, state, action, {name: value})
                )
            else:
                wider.append(compiled)

        tables: list[SupportTable] = []
        constraints: list[ScopedConstraint] = []
        for compiled in wider:
            if len(compiled.arguments) == 2:
                first, second = compiled.arguments
                allowed = frozenset(
                    (first_value, second_value)
                    for first_value in domains[first]
                    for second_value in domains[second]
                    if self._constraint_checker.holds(
                        compiled, state, action, {first: first_value, second: second_value}
                    )
                )
                tables.append(SupportTable(first, second, allowed))
            else:
                constraints.append(ScopedConstraint(compiled, compiled.arguments))

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

    def _group(
        self, constraint: PythonRule, names: list[str], definitions: PythonRule | None
    ) -> tuple[list[str], list[CompiledRule]] | None:
        """The parameters and the parameter-free operands of an all_different call, or None when it isn't one or an
        operand is anything else."""
        operands = self._call_operand_mapper.to_operands(constraint, ALL_DIFFERENT)
        if operands is None:
            return None
        parameters: list[str] = []
        fixed: list[CompiledRule] = []
        for operand in operands:
            if operand.source in names:
                parameters.append(operand.source)
                continue
            compiled = self._rule_compiler.compile_value(operand, names, definitions)
            if compiled.arguments:
                return None
            fixed.append(compiled)
        return parameters, fixed

    def _no_solution(self, action: str, constraint: PythonRule) -> tuple[list[Action], SolveStatistics]:
        logger.debug("%s: no solution, constraint is false: %s", action, constraint.source)
        return [], SolveStatistics(0, 0, 0, 0)
