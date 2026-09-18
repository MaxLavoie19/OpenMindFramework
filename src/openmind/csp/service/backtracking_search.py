import logging
from collections.abc import Iterable

from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.csp.model.wipeout import Wipeout
from openmind.csp.service.all_different_propagator import AllDifferentPropagator
from openmind.csp.service.arc_consistency import ArcConsistency
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.world.model.state import State
from openmind.world.model.value import Value

logger = logging.getLogger(__name__)

type Domains = dict[str, frozenset[Value]]


class BacktrackingSearch:
    """Backtracking that maintains arc consistency and all-different filtering after every assignment, forward-checks
    larger constraints, picks the variable with the fewest values left (then the most constraints), and tries the
    least-constraining values first when a limit is set."""

    def __init__(
        self,
        arc_consistency: ArcConsistency,
        all_different_propagator: AllDifferentPropagator,
        constraint_checker: ConstraintChecker,
    ) -> None:
        self._arc_consistency = arc_consistency
        self._all_different_propagator = all_different_propagator
        self._constraint_checker = constraint_checker

    def search(
        self, space: SearchSpace, state: State, limit: int | None
    ) -> tuple[tuple[dict[str, Value], ...], SolveStatistics]:
        """Every solution, or at most limit, as parameter values, with what the search did."""
        names = [name for name, _ in space.variables]
        position = {name: index for index, name in enumerate(names)}
        order = {name: values for name, values in space.variables}
        degree = dict.fromkeys(names, 0)
        for table in space.tables:
            degree[table.first] += 1
            degree[table.second] += 1
        for group in space.groups:
            for name in group.variables:
                degree[name] += 1
        for constraint in space.constraints:
            for name in constraint.scope:
                degree[name] += 1

        domains: Domains = {name: frozenset(values) for name, values in space.variables}
        size = sum(map(len, domains.values()))
        try:
            for name in names:
                if not domains[name]:
                    raise Wipeout(name)
            domains = self._propagate(space, state, domains, names)
        except Wipeout as wipeout:
            logger.debug("%s: no solution, %s has no value left", space.action, wipeout.variable)
            return (), SolveStatistics(0, 0, 0, 0)

        solutions: list[dict[str, Value]] = []
        assignments = 0
        dead_ends = 0
        pruned = size - sum(map(len, domains.values()))

        def backtrack(domains: Domains) -> bool:
            nonlocal assignments, dead_ends, pruned
            open_variables = [name for name in names if len(domains[name]) > 1]
            if not open_variables:
                solutions.append({name: next(iter(domains[name])) for name in names})
                return limit is not None and len(solutions) >= limit
            variable = min(open_variables, key=lambda name: (len(domains[name]), -degree[name], position[name]))
            for value in self._values(space, domains, variable, order[variable], limit):
                assignments += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("%s: try %s = %r", space.action, variable, value)
                narrowed = dict(domains)
                narrowed[variable] = frozenset((value,))
                try:
                    propagated = self._propagate(space, state, narrowed, (variable,))
                except Wipeout as wipeout:
                    dead_ends += 1
                    logger.debug(
                        "%s: dead end at %s = %r, %s has no value left",
                        space.action,
                        variable,
                        value,
                        wipeout.variable,
                    )
                    continue
                pruned += sum(map(len, narrowed.values())) - sum(map(len, propagated.values()))
                if backtrack(propagated):
                    return True
            return False

        backtrack(domains)
        return tuple(solutions), SolveStatistics(len(solutions), assignments, dead_ends, pruned)

    def _propagate(self, space: SearchSpace, state: State, domains: Domains, changed: Iterable[str]) -> Domains:
        """Runs arc consistency, all-different filtering and forward checking until nothing changes."""
        pending = set(changed)
        while pending:
            before = domains
            domains = self._arc_consistency.propagate(domains, space.tables, pending)
            touched = pending | {name for name, values in domains.items() if values is not before[name]}
            pending = set()
            for group in space.groups:
                if touched.isdisjoint(group.variables):
                    continue
                narrowed = self._all_different_propagator.propagate(domains, group)
                pending.update(name for name in group.variables if narrowed[name] is not domains[name])
                domains = narrowed
            for constraint in space.constraints:
                if touched.isdisjoint(constraint.scope):
                    continue
                narrowed = self._forward_check(space.action, state, domains, constraint)
                pending.update(name for name in constraint.scope if narrowed[name] is not domains[name])
                domains = narrowed
        return domains

    def _forward_check(self, action: str, state: State, domains: Domains, constraint: ScopedConstraint) -> Domains:
        open_variables = [name for name in constraint.scope if len(domains[name]) > 1]
        if len(open_variables) > 1:
            return domains
        fixed = {name: next(iter(domains[name])) for name in constraint.scope if len(domains[name]) == 1}
        if not open_variables:
            if self._constraint_checker.holds(constraint.rule, state, action, fixed):
                return domains
            raise Wipeout(constraint.scope[-1])
        (target,) = open_variables
        kept = frozenset(
            value
            for value in domains[target]
            if self._constraint_checker.holds(constraint.rule, state, action, fixed | {target: value})
        )
        if not kept:
            raise Wipeout(target)
        if len(kept) == len(domains[target]):
            return domains
        narrowed = dict(domains)
        narrowed[target] = kept
        return narrowed

    def _values(
        self,
        space: SearchSpace,
        domains: Domains,
        variable: str,
        order: tuple[Value, ...],
        limit: int | None,
    ) -> list[Value]:
        candidates = [value for value in order if value in domains[variable]]
        if limit is None:
            return candidates
        return sorted(candidates, key=lambda value: self._eliminations(space, domains, variable, value))

    def _eliminations(self, space: SearchSpace, domains: Domains, variable: str, value: Value) -> int:
        """How many values assigning this value would rule out in the other variables, by tables and groups."""
        count = 0
        for table in space.tables:
            if table.first == variable:
                count += sum(1 for partner in domains[table.second] if (value, partner) not in table.allowed)
            elif table.second == variable:
                count += sum(1 for partner in domains[table.first] if (partner, value) not in table.allowed)
        for group in space.groups:
            if variable in group.variables:
                count += sum(1 for other in group.variables if other != variable and value in domains[other])
        return count
