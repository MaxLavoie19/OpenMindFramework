import logging
from operator import itemgetter

from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.csp.model.state_domain import StateDomain
from openmind.csp.model.support_table import SupportTable
from openmind.csp.model.variable import Variable
from openmind.csp.service.backtracking_search import BacktrackingSearch
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.rule.constant.rule_constant import ALL_DIFFERENT
from openmind.rule.mapper.call_operand_mapper import CallOperandMapper
from openmind.rule.model.compiled_rule import CompiledRule
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.constant.players_constant import PLAYER
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value

logger = logging.getLogger(__name__)


class Solver:
    """Finds the actions whose parameter values satisfy all of their constraints in a state. The constraints reading no
    parameter are checked first; then each parameter gets its values, fixed or computed from the state; each other
    constraint, a Python rule, takes the strongest form the parameters it reads allow (a domain filter, an all-different
    group, a support table, or a forward-checked constraint), backtracking search does the rest, and results are cached
    per problem and state until the process's memory guard clears them."""

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
        self._cache: dict[tuple[int, State, int | None], tuple[Problem, tuple[tuple[Action, ...], SolveStatistics]]] = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def __getstate__(self) -> dict[str, object]:
        """The cache stays behind when the solver is copied to another process: its keys are this process's ids."""
        return {name: value for name, value in self.__dict__.items() if name not in ("_cache", "_memory_guard")}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._cache = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def memory_entries(self) -> int:
        return len(self._cache)

    def clear_memory(self) -> None:
        self._cache.clear()

    def solve(
        self, problem: Problem, state: State, limit: int | None = None, player: str | None = None
    ) -> tuple[Action, ...]:
        """Every solution, or at most limit, ordered by the variables' domains with the first variable changing slowest.
        Given a player, such as one of several players acting at once, the rules also read it as `player`."""
        return self.solve_with_statistics(problem, state, limit, player)[0]

    def solve_with_statistics(
        self, problem: Problem, state: State, limit: int | None = None, player: str | None = None
    ) -> tuple[tuple[Action, ...], SolveStatistics]:
        """The solutions solve gives, with what the search did summed over the problem's actions. A cached result keeps
        the statistics of the search that found it. Given a player, a state variable named `player` raises ValueError."""
        if player is not None:
            if any(name == PLAYER for name, _ in state.variables):
                raise ValueError(f"A state variable is named {PLAYER!r}, the name rules read the player solved for by")
            state = State((*state.variables, (PLAYER, player)))
        key = (id(problem), state, limit)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is problem:
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
        self._memory_guard.remembered()
        self._cache[key] = (problem, result)
        return result

    def _solve_action(
        self, definition: ActionDefinition, state: State, limit: int | None, definitions: PythonRule | None
    ) -> tuple[list[Action], SolveStatistics]:
        action = definition.name
        names = [variable.name for variable in definition.variables]
        compiled_constraints = [
            (constraint, self._rule_compiler.compile_value(constraint, names, definitions))
            for constraint in definition.constraints
        ]
        for constraint, compiled in compiled_constraints:
            if not compiled.arguments and not self._constraint_checker.holds(compiled, state, action, {}):
                return self._no_solution(action, constraint)
        ordered = {variable.name: self._values(variable.domain, state, definitions) for variable in definition.variables}
        domains = dict(ordered)

        groups: list[AllDifferentGroup] = []
        wider: list[CompiledRule] = []
        for constraint, compiled in compiled_constraints:
            scope = compiled.arguments
            if not scope:
                continue
            group = self._group(constraint, names, definitions)
            if group is not None:
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
        position = {name: {value: index for index, value in enumerate(ordered[name])} for name in names}
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

    def _values(
        self, domain: DiscreteDomain | StateDomain, state: State, definitions: PythonRule | None
    ) -> tuple[Value, ...]:
        """A discrete domain's values, or the values a state domain's rule gives in the state, each once, in order."""
        if isinstance(domain, DiscreteDomain):
            return domain.values
        compiled = self._rule_compiler.compile_value(domain.rule, (), definitions)
        return tuple(dict.fromkeys(self._rule_runner.value(compiled, state)))  # type: ignore[call-overload]

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
