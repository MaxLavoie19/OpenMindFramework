import logging
from collections.abc import Mapping, Sequence
from operator import itemgetter

from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.circuit_constraint import CircuitConstraint
from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.csp.model.support_table import SupportTable
from openmind.csp.service.backtracking_search import BacktrackingSearch
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.csp.repository.solution_cache import SolutionCache
from openmind.rule.constant.rule_constant import ALL_DIFFERENT, CIRCUIT
from openmind.rule.mapper.call_operand_mapper import CallOperandMapper
from openmind.rule.model.called_rule import CalledRule
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.constant.players_constant import PLAYER
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)


class Solver:
    """Finds the actions whose parameter values satisfy all of their constraints in a state.

    It solves over rules: an action is a name, a values rule per parameter and its constraint rules, which is what an
    RBS hands it (see `rbs/README.md`). The constraints reading no parameter are checked first; then each parameter's
    rule gives its values; each other constraint takes the strongest form the parameters it reads allow (a domain
    filter, an all-different group, a support table, or a forward-checked constraint), backtracking search does the
    rest, and results are kept per action and state in the solution cache it is given."""

    def __init__(
        self,
        rule_caller: RuleCaller,
        call_operand_mapper: CallOperandMapper,
        constraint_checker: ConstraintChecker,
        backtracking_search: BacktrackingSearch,
        solution_cache: SolutionCache,
    ) -> None:
        self._rule_caller = rule_caller
        self._call_operand_mapper = call_operand_mapper
        self._constraint_checker = constraint_checker
        self._backtracking_search = backtracking_search
        self._cache = solution_cache

    def solve(
        self,
        state: State,
        action: str,
        values: Mapping[str, Rule],
        constraints: Sequence[Rule] = (),
        definitions: PythonRule | None = None,
        limit: int | None = None,
        player: str | None = None,
        value_order: str | None = None,
    ) -> tuple[Action, ...]:
        """Every action of that name whose parameter values satisfy its constraints, or at most limit, ordered by the
        parameters' values with the first parameter changing slowest. Given a player, such as one of several players
        acting at once, the rules also read it as `player`.

        `value_order` says which value a variable is tried with first; left out, it is the least constraining
        wherever a limit is set, which is what solving for one solution has always done."""
        return self.solve_with_statistics(
            state, action, values, constraints, definitions, limit, player, value_order
        )[0]

    def allows(
        self,
        state: State,
        action: Action,
        values: Mapping[str, Rule],
        constraints: Sequence[Rule] = (),
        definitions: PythonRule | None = None,
        player: str | None = None,
    ) -> bool:
        """Whether that one action is legal in the state: its constraints run against the values it already carries,
        rather than searched for. An action a model computed isn't legal by construction, so whatever computes one —
        equations, a controller, a model proposing a solution — has what it gives checked here.

        A parameter the action doesn't carry, or a value its rule doesn't allow, makes it illegal."""
        if player is not None:
            if state.has(PLAYER):
                raise ValueError(f"A state model is named {PLAYER!r}, the name rules read the player solved for by")
            state = state.with_model(PLAYER, player)
        carried = dict(action.parameters)
        if set(carried) != set(values):
            logger.debug("%s carries %s, not %s", action.name, sorted(carried), sorted(values))
            return False
        for name, rule in values.items():
            if carried[name] not in self._values(rule, state, definitions):
                logger.debug("%s is not a value %s can take in %s", carried[name], name, action.name)
                return False
        names = tuple(values)
        for constraint in constraints:
            prepared = self._rule_caller.prepare(constraint, names, definitions)
            if not self._constraint_checker.holds(prepared, state, action.name, carried):
                logger.debug("%s is refused by %s", action.name, prepared.source)
                return False
        return True

    def solve_with_statistics(
        self,
        state: State,
        action: str,
        values: Mapping[str, Rule],
        constraints: Sequence[Rule] = (),
        definitions: PythonRule | None = None,
        limit: int | None = None,
        player: str | None = None,
        value_order: str | None = None,
    ) -> tuple[tuple[Action, ...], SolveStatistics]:
        """The solutions solve gives, with what the search did. A cached result keeps the statistics of the search that
        found it. Given a player, a state model named `player` raises ValueError."""
        if player is not None:
            if state.has(PLAYER):
                raise ValueError(f"A state model is named {PLAYER!r}, the name rules read the player solved for by")
            state = state.with_model(PLAYER, player)
        rules = (action, tuple(values.items()), tuple(constraints))
        key = (state, limit, rules, value_order)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        found, statistics = self._solve_action(
            action, values, constraints, state, limit, definitions, value_order
        )
        return self._cache.put(key, (tuple(found), statistics))

    def _solve_action(
        self,
        action: str,
        values: Mapping[str, Rule],
        constraints: Sequence[Rule],
        state: State,
        limit: int | None,
        definitions: PythonRule | None,
        value_order: str | None = None,
    ) -> tuple[list[Action], SolveStatistics]:
        names = list(values)
        named = set(names)
        prepared_constraints = [
            (constraint, self._rule_caller.prepare(constraint, names, definitions)) for constraint in constraints
        ]
        for constraint, prepared in prepared_constraints:
            if not prepared.arguments and not self._constraint_checker.holds(prepared, state, action, {}):
                return self._no_solution(action, constraint)
        ordered = {name: self._values(rule, state, definitions) for name, rule in values.items()}
        domains = dict(ordered)

        groups: list[AllDifferentGroup] = []
        circuits: list[CircuitConstraint] = []
        wider: list[CalledRule] = []
        for constraint, prepared in prepared_constraints:
            scope = prepared.arguments
            if not scope:
                continue
            circuit = self._circuit(constraint, named)
            if circuit is not None:
                self._check_positions(action, circuit, domains)
                if len(circuit) > 1:
                    # No position follows itself, which is the one thing a circuit refuses outright rather than
                    # by propagation. Taken out here, as a domain filter, it is paid once instead of at every node.
                    for place, name in enumerate(circuit):
                        domains[name] = tuple(value for value in domains[name] if value != place)
                circuits.append(CircuitConstraint(circuit))
                continue
            group = self._group(constraint, names, named, definitions)
            if group is not None:
                parameters, fixed_rules = group
                fixed = [self._rule_caller.call(rule, state) for rule in fixed_rules]
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
                    if self._constraint_checker.holds(prepared, state, action, {name: value})
                )
            else:
                wider.append(prepared)

        tables: list[SupportTable] = []
        constraints: list[ScopedConstraint] = []
        for prepared in wider:
            if len(prepared.arguments) == 2:
                first, second = prepared.arguments
                allowed = frozenset(
                    (first_value, second_value)
                    for first_value in domains[first]
                    for second_value in domains[second]
                    if self._constraint_checker.holds(
                        prepared, state, action, {first: first_value, second: second_value}
                    )
                )
                tables.append(SupportTable(first, second, allowed))
            else:
                constraints.append(ScopedConstraint(prepared, prepared.arguments))

        space = SearchSpace(
            action,
            tuple((name, domains[name]) for name in names),
            tuple(tables),
            tuple(groups),
            tuple(constraints),
            tuple(circuits),
        )
        solutions, statistics = self._backtracking_search.search(space, state, limit, value_order)
        if len(solutions) > 1:
            # The map from a value to where its rule gave it costs one entry per value of every parameter, which
            # is the size of the whole problem. There is nothing to order with fewer than two solutions.
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

    def _values(self, rule: Rule, state: State, definitions: PythonRule | None) -> tuple[Value, ...]:
        """The values the parameter's rule gives in the state, each once, in the order it gives them. A rule that reads
        nothing gives the same values in every state."""
        return tuple(dict.fromkeys(self._rule_caller.value(rule, state, None, None, definitions)))  # type: ignore[call-overload]

    def _group(
        self, constraint: Rule, names: list[str], named: set[str], definitions: PythonRule | None
    ) -> tuple[list[str], list[CalledRule]] | None:
        """The parameters and the parameter-free operands of an all_different call, or None when it isn't one, an operand
        is anything else, or the constraint is one of the project's functions, whose operands can't be read.

        `named` is `names` as a set: a call over every parameter of the action has as many operands as there are
        parameters, so looking each one up in a list costs the square of them."""
        if not isinstance(constraint, PythonRule):
            return None
        operands = self._call_operand_mapper.to_operands(constraint, ALL_DIFFERENT)
        if operands is None:
            return None
        parameters: list[str] = []
        fixed: list[CalledRule] = []
        for operand in operands:
            if operand.source in named:
                parameters.append(operand.source)
                continue
            prepared = self._rule_caller.prepare(operand, names, definitions)
            if prepared.arguments:
                return None
            fixed.append(prepared)
        return parameters, fixed

    def _circuit(self, constraint: Rule, named: set[str]) -> tuple[str, ...] | None:
        """The parameters of a circuit call, in the order it names them, or None when the constraint isn't one.

        **The operand order is the numbering**, and never `CalledRule.arguments`, which is the order the parameters
        were declared in: all_different has no use for an order and a circuit is nothing but one. A value `j` means
        the position standing `j`th here."""
        if not isinstance(constraint, PythonRule):
            return None
        operands = self._call_operand_mapper.to_operands(constraint, CIRCUIT)
        if operands is None:
            return None
        variables = tuple(operand.source for operand in operands)
        unknown = [source for source in variables if source not in named]
        if unknown:
            raise ValueError(
                f"{CIRCUIT} takes the action's parameters, and a successor it fixes outright is not supported: "
                f"{unknown[0]!r}"
            )
        if len(set(variables)) < len(variables):
            repeated = next(source for source in variables if variables.count(source) > 1)
            raise ValueError(f"{CIRCUIT} names {repeated!r} more than once, so it has no position of its own")
        return variables

    def _check_positions(self, action: str, variables: tuple[str, ...], domains: dict[str, tuple[Value, ...]]) -> None:
        """Every value a circuit's parameter can take names one of its positions, and nothing else can be meant."""
        for name in variables:
            for value in domains[name]:
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(variables):
                    raise ValueError(
                        f"{action}: {name} can take {value!r}, which names no position of a {CIRCUIT} over "
                        f"{len(variables)} parameters"
                    )

    def _no_solution(self, action: str, constraint: Rule) -> tuple[list[Action], SolveStatistics]:
        logger.debug("%s: no solution, constraint is false: %s", action, self._rule_caller.source(constraint))
        return [], SolveStatistics(0, 0, 0, 0)
