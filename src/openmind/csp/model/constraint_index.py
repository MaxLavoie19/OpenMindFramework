from collections.abc import Iterable
from dataclasses import dataclass

from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.circuit_constraint import CircuitConstraint
from openmind.csp.model.scoped_constraint import ScopedConstraint
from openmind.csp.model.search_space import SearchSpace
from openmind.csp.model.support_table import SupportTable
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class ConstraintIndex:
    """Which constraints each variable is in, worked out once for a search.

    **Propagation revisits what a change can reach, not everything there is.** Asking every constraint whether it
    holds any of the variables that changed costs the sum of every scope, every round — which one constraint over
    every variable makes a full pass by itself. Asked this way it costs what changed.

    `holders` answers the same question values ask: which positions of a circuit could still take this value, which
    is what tells the permutation filter where to look and what tells the least-constraining count what a value
    would cost."""

    tables: tuple[SupportTable, ...]
    groups: tuple[AllDifferentGroup, ...]
    circuits: tuple[CircuitConstraint, ...]
    constraints: tuple[ScopedConstraint, ...]
    _tables_of: dict[str, tuple[int, ...]]
    _groups_of: dict[str, tuple[int, ...]]
    _circuits_of: dict[str, tuple[int, ...]]
    _constraints_of: dict[str, tuple[int, ...]]
    _holders: tuple[dict[Value, tuple[str, ...]], ...]
    _positions: tuple[dict[str, int], ...]

    @staticmethod
    def of(space: SearchSpace) -> "ConstraintIndex":
        domains = dict(space.variables)
        tables_of: dict[str, list[int]] = {}
        groups_of: dict[str, list[int]] = {}
        circuits_of: dict[str, list[int]] = {}
        constraints_of: dict[str, list[int]] = {}
        for place, table in enumerate(space.tables):
            tables_of.setdefault(table.first, []).append(place)
            tables_of.setdefault(table.second, []).append(place)
        for place, group in enumerate(space.groups):
            for name in group.variables:
                groups_of.setdefault(name, []).append(place)
        for place, circuit in enumerate(space.circuits):
            for name in circuit.variables:
                circuits_of.setdefault(name, []).append(place)
        for place, constraint in enumerate(space.constraints):
            for name in constraint.scope:
                constraints_of.setdefault(name, []).append(place)

        holders = []
        for circuit in space.circuits:
            held: dict[Value, list[str]] = {}
            for name in circuit.variables:
                for value in domains.get(name, ()):
                    held.setdefault(value, []).append(name)
            holders.append({value: tuple(names) for value, names in held.items()})

        return ConstraintIndex(
            space.tables,
            space.groups,
            space.circuits,
            space.constraints,
            {name: tuple(places) for name, places in tables_of.items()},
            {name: tuple(places) for name, places in groups_of.items()},
            {name: tuple(places) for name, places in circuits_of.items()},
            {name: tuple(places) for name, places in constraints_of.items()},
            tuple(holders),
            tuple({name: place for place, name in enumerate(circuit.variables)} for circuit in space.circuits),
        )

    def tables_of(self, variable: str) -> tuple[SupportTable, ...]:
        return tuple(self.tables[place] for place in self._tables_of.get(variable, ()))

    def groups_of(self, variable: str) -> tuple[AllDifferentGroup, ...]:
        return tuple(self.groups[place] for place in self._groups_of.get(variable, ()))

    def circuit_places_of(self, variable: str) -> tuple[int, ...]:
        """Where each circuit holding that variable sits. Circuits are named by their place and never by value: a
        circuit over a hundred thousand parameters is a hundred-thousand-element tuple, and comparing or hashing
        one to look it up would cost as much as the propagation it was looked up for."""
        return self._circuits_of.get(variable, ())

    def groups_touching(self, variables: Iterable[str]) -> tuple[AllDifferentGroup, ...]:
        return tuple(self.groups[place] for place in self._touched(self._groups_of, variables))

    def circuit_places_touching(self, variables: Iterable[str]) -> tuple[int, ...]:
        return tuple(self._touched(self._circuits_of, variables))

    def constraints_touching(self, variables: Iterable[str]) -> tuple[ScopedConstraint, ...]:
        return tuple(self.constraints[place] for place in self._touched(self._constraints_of, variables))

    def holders(self, place: int, value: Value) -> tuple[str, ...]:
        """The variables of the circuit at that place whose values rule offered this value."""
        return self._holders[place].get(value, ())

    def positions(self, place: int) -> dict[str, int]:
        """Where each variable of the circuit at that place stands in it, which is the number its value names it
        by."""
        return self._positions[place]

    def _touched(self, places_of: dict[str, tuple[int, ...]], variables: Iterable[str]) -> list[int]:
        found: dict[int, None] = {}
        for variable in variables:
            for place in places_of.get(variable, ()):
                found[place] = None
        return sorted(found)
