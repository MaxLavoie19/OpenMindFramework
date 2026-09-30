import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from openmind.csp.constant.solver_constant import DOMAIN_ORDER, LEAST_CONSTRAINING
from openmind.csp.model.constraint_index import ConstraintIndex
from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.csp.model.wipeout import Wipeout
from openmind.csp.repository.circuit_chains import CircuitChains
from openmind.csp.repository.domain_repository import DomainRepository
from openmind.csp.repository.variable_queue import VariableQueue
from openmind.csp.service.all_different_propagator import AllDifferentPropagator
from openmind.csp.service.arc_consistency import ArcConsistency
from openmind.csp.service.circuit_propagator import CircuitPropagator
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.world.model.state import State
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Frame:
    """One variable being decided: the values left to try it with, and where to unwind the trail back to."""

    variable: str
    candidates: list[Value]
    mark: tuple[int, tuple[int, ...]]
    tried: int = field(default=0)


class BacktrackingSearch:
    """Backtracking that maintains arc consistency and all-different filtering after every assignment, forward-checks
    larger constraints, picks the variable with the fewest values left (then the most constraints), and tries the
    least-constraining values first when a limit is set.

    **A node costs what it changed.** The domains live in one repository with a trail, so deciding a variable is a
    mark and an assignment rather than a copy of every domain, and undoing one is putting back the few values that
    were taken. Nothing here is kept between calls: the repository, the queue and the index are made for the search
    and handed to the propagators, which keep nothing either.

    **And the frames are a list, not the interpreter's stack.** A search is as deep as it has variables, which a
    puzzle of fifty never noticed and a problem of a hundred thousand cannot survive."""

    def __init__(
        self,
        arc_consistency: ArcConsistency,
        all_different_propagator: AllDifferentPropagator,
        circuit_propagator: CircuitPropagator,
        constraint_checker: ConstraintChecker,
    ) -> None:
        self._arc_consistency = arc_consistency
        self._all_different_propagator = all_different_propagator
        self._circuit_propagator = circuit_propagator
        self._constraint_checker = constraint_checker

    def search(
        self,
        space: SearchSpace,
        state: State,
        limit: int | None,
        value_order: str | None = None,
    ) -> tuple[tuple[dict[str, Value], ...], SolveStatistics]:
        """Every solution, or at most limit, as parameter values, with what the search did.

        `value_order` says which values a variable is tried with first: in the order its rule gave them, or the
        least constraining first. Left out, it is the least constraining wherever a limit is set, which is what a
        search for one solution has always done."""
        names = [name for name, _ in space.variables]
        position = {name: place for place, name in enumerate(names)}
        order = dict(space.variables)
        degree = self._degrees(space, names)
        index = ConstraintIndex.of(space)
        repository = DomainRepository(space.variables)
        chains = tuple(CircuitChains(len(circuit.variables)) for circuit in space.circuits)
        queue = VariableQueue(degree, position)
        for name in names:
            queue.push(name, len(repository.values[name]))
        if value_order is None:
            value_order = LEAST_CONSTRAINING if limit is not None else DOMAIN_ORDER

        try:
            for name in names:
                if not repository.values[name]:
                    raise Wipeout(name)
            self._propagate(space, index, state, repository, queue, chains, names)
        except Wipeout as wipeout:
            logger.debug("%s: no solution, %s has no value left", space.action, wipeout.variable)
            return (), SolveStatistics(0, 0, 0, 0)

        solutions: list[dict[str, Value]] = []
        assignments = 0
        dead_ends = 0
        pruned = repository.height
        frames: list[_Frame] = []

        while True:
            variable = queue.pick(repository)
            if variable is None:
                solutions.append({name: repository.only(name) for name in names})
                if limit is not None and len(solutions) >= limit:
                    break
            else:
                frames.append(
                    _Frame(
                        variable,
                        self._candidates(index, repository, variable, order[variable], value_order),
                        self._mark(repository, chains),
                    )
                )

            advanced = False
            while frames:
                frame = frames[-1]
                tried, ended, taken = self._advance(space, index, state, repository, queue, chains, frame)
                assignments += tried
                dead_ends += ended
                if taken is not None:
                    pruned += taken
                    advanced = True
                    break
                self._retreat(repository, queue, chains, frame.mark)
                frames.pop()
            if not advanced:
                break

        return tuple(solutions), SolveStatistics(len(solutions), assignments, dead_ends, pruned)

    def _advance(
        self,
        space: SearchSpace,
        index: ConstraintIndex,
        state: State,
        repository: DomainRepository,
        queue: VariableQueue,
        chains: tuple[CircuitChains, ...],
        frame: _Frame,
    ) -> tuple[int, int, int | None]:
        """Tries the frame's values until one propagates. Gives back how many were tried, how many were dead ends,
        and how many values propagation removed — or None where the frame has nothing left."""
        tried = 0
        dead_ends = 0
        while frame.tried < len(frame.candidates):
            value = frame.candidates[frame.tried]
            frame.tried += 1
            tried += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("%s: try %s = %r", space.action, frame.variable, value)
            self._retreat(repository, queue, chains, frame.mark)
            try:
                repository.fix(frame.variable, value)
                settled = repository.height
                self._propagate(space, index, state, repository, queue, chains, (frame.variable,))
            except Wipeout as wipeout:
                dead_ends += 1
                logger.debug(
                    "%s: dead end at %s = %r, %s has no value left",
                    space.action,
                    frame.variable,
                    value,
                    wipeout.variable,
                )
                continue
            return tried, dead_ends, repository.height - settled
        return tried, dead_ends, None

    def _mark(self, repository: DomainRepository, chains: tuple[CircuitChains, ...]) -> tuple[int, tuple[int, ...]]:
        """Where every trail stands, which is what a frame unwinds back to."""
        return repository.height, tuple(chain.height for chain in chains)

    def _retreat(
        self,
        repository: DomainRepository,
        queue: VariableQueue,
        chains: tuple[CircuitChains, ...],
        mark: tuple[int, tuple[int, ...]],
    ) -> None:
        """Puts every trail back as it stood at the mark, and offers every variable that grew again."""
        domains, recorded = mark
        for restored in repository.undo_to(domains):
            queue.push(restored, len(repository.values[restored]))
        for chain, height in zip(chains, recorded):
            chain.undo_to(height)

    def _propagate(
        self,
        space: SearchSpace,
        index: ConstraintIndex,
        state: State,
        repository: DomainRepository,
        queue: VariableQueue,
        chains: tuple[CircuitChains, ...],
        changed: Iterable[str],
    ) -> None:
        """Runs arc consistency, all-different filtering and forward checking until nothing changes."""
        entry = repository.height
        pending = set(changed)
        while pending:
            mark = repository.height
            self._arc_consistency.propagate(repository, index, pending)
            touched = pending | set(repository.changed_since(mark))
            pending = set()
            for group in index.groups_touching(touched):
                before = repository.height
                self._all_different_propagator.propagate(repository, group)
                pending.update(repository.changed_since(before))
            for place in index.circuit_places_touching(touched):
                before = repository.height
                self._circuit_propagator.propagate(repository, index, place, chains[place], touched)
                pending.update(repository.changed_since(before))
            for constraint in index.constraints_touching(touched):
                before = repository.height
                self._forward_check(space.action, state, repository, constraint)
                pending.update(repository.changed_since(before))
        for name in repository.changed_since(entry):
            queue.push(name, len(repository.values[name]))

    def _forward_check(
        self, action: str, state: State, repository: DomainRepository, constraint: ScopedConstraint
    ) -> None:
        open_variables = [name for name in constraint.scope if not repository.settled(name)]
        if len(open_variables) > 1:
            return
        fixed = {name: repository.only(name) for name in constraint.scope if repository.settled(name)}
        if not open_variables:
            if self._constraint_checker.holds(constraint.rule, state, action, fixed):
                return
            raise Wipeout(constraint.scope[-1])
        (target,) = open_variables
        dropped = [
            value
            for value in repository.values[target]
            if not self._constraint_checker.holds(constraint.rule, state, action, fixed | {target: value})
        ]
        for value in dropped:
            repository.remove(target, value)

    def _degrees(self, space: SearchSpace, names: list[str]) -> dict[str, int]:
        degree = dict.fromkeys(names, 0)
        for table in space.tables:
            degree[table.first] += 1
            degree[table.second] += 1
        for group in space.groups:
            for name in group.variables:
                degree[name] += 1
        for circuit in space.circuits:
            for name in circuit.variables:
                degree[name] += 1
        for constraint in space.constraints:
            for name in constraint.scope:
                degree[name] += 1
        return degree

    def _candidates(
        self,
        index: ConstraintIndex,
        repository: DomainRepository,
        variable: str,
        order: tuple[Value, ...],
        value_order: str,
    ) -> list[Value]:
        held = repository.values[variable]
        candidates = [value for value in order if value in held]
        if value_order != LEAST_CONSTRAINING:
            return candidates
        return sorted(candidates, key=lambda value: self._eliminations(index, repository, variable, value))

    def _eliminations(
        self, index: ConstraintIndex, repository: DomainRepository, variable: str, value: Value
    ) -> int:
        """How many values assigning this value would rule out in the other variables, by the constraints this one
        is in — never by every constraint there is."""
        count = 0
        for table in index.tables_of(variable):
            if table.first == variable:
                count += sum(
                    1 for partner in repository.values[table.second] if (value, partner) not in table.allowed
                )
            elif table.second == variable:
                count += sum(
                    1 for partner in repository.values[table.first] if (partner, value) not in table.allowed
                )
        for group in index.groups_of(variable):
            count += sum(
                1 for other in group.variables if other != variable and value in repository.values[other]
            )
        for place in index.circuit_places_of(variable):
            count += sum(
                1 for other in index.holders(place, value) if other != variable and value in repository.values[other]
            )
        return count
