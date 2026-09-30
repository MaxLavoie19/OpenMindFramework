from collections.abc import Iterable

from openmind.csp.model.constraint_index import ConstraintIndex
from openmind.csp.model.wipeout import Wipeout
from openmind.csp.repository.circuit_chains import CircuitChains
from openmind.csp.repository.domain_repository import DomainRepository


class CircuitPropagator:
    """Keeps a circuit's variables making one cycle through every position rather than several loops.

    Two duties, and the first is why a circuit needs no all-different beside it:

    **The successors are all different.** A position decided on `j` has `j` taken out of every other position that
    could have had it, which costs the number of positions that offered the value rather than the width of the
    whole constraint.

    **And they make one cycle.** Deciding `i` is followed by `j` joins the run ending at `i` to the run starting at
    `j`; the joined run's own start is then refused at its end for as long as the run is short of covering
    everything, and required there the moment it does. That is what stops a set of small loops, which is the whole
    difference between this and all-different.

    A domain narrowed to one value is a decision that no one ever tries, so what this reads is which positions are
    settled — never which were assigned."""

    def propagate(
        self,
        repository: DomainRepository,
        index: ConstraintIndex,
        place: int,
        chains: CircuitChains,
        changed: Iterable[str],
    ) -> None:
        """Narrows the domains of the circuit at that place, from the variables that changed. Raises Wipeout where
        a position is left with no successor, or where the cycle would close short."""
        constraint = index.circuits[place]
        variables = constraint.variables
        positions = index.positions(place)
        total = len(variables)

        for name in changed:
            position = positions.get(name)
            if position is None or not repository.settled(name):
                continue
            successor = repository.only(name)
            if chains.recorded(position) is not None:
                continue
            chains.mark(position, successor)

            for other in index.holders(place, successor):
                if other != name:
                    repository.remove(other, successor)

            if chains.start_of(position) == successor:
                if chains.length_of(successor) != total:
                    raise Wipeout(name)
                continue

            start, end, length = chains.join(position, successor)
            if length < total:
                repository.remove(variables[end], start)
                continue
            closing = variables[end]
            for value in [held for held in repository.values[closing] if held != start]:
                repository.remove(closing, value)
