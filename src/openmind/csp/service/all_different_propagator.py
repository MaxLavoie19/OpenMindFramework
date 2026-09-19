from collections import deque

from openmind.csp.model.all_different_group import AllDifferentGroup
from openmind.csp.model.wipeout import Wipeout
from openmind.structure.model.value import Value

type Node = tuple[str, object]


class AllDifferentPropagator:
    """Régin's filtering for all-different: keeps exactly the values that belong to some assignment giving every variable
    of the group a different value."""

    def propagate(
        self, domains: dict[str, frozenset[Value]], group: AllDifferentGroup
    ) -> dict[str, frozenset[Value]]:
        """Narrows the group's domains; unchanged domains are kept as they are. Raises Wipeout when the variables can't
        all take different values."""
        matching = self._maximum_matching(domains, group.variables)
        edges: dict[Node, list[Node]] = {}
        values: set[Value] = set()
        for variable in group.variables:
            edges.setdefault(("variable", variable), []).append(("value", matching[variable]))
            for value in domains[variable]:
                values.add(value)
                if value != matching[variable]:
                    edges.setdefault(("value", value), []).append(("variable", variable))
        matched = set(matching.values())
        reachable = self._reachable(edges, [("value", value) for value in values if value not in matched])
        nodes = [("variable", variable) for variable in group.variables] + [("value", value) for value in values]
        component = self._components(edges, nodes)

        narrowed = dict(domains)
        for variable in group.variables:
            kept = frozenset(
                value
                for value in domains[variable]
                if value == matching[variable]
                or component[("value", value)] == component[("variable", variable)]
                or ("value", value) in reachable
            )
            if len(kept) < len(domains[variable]):
                narrowed[variable] = kept
        return narrowed

    def _maximum_matching(self, domains: dict[str, frozenset[Value]], variables: tuple[str, ...]) -> dict[str, Value]:
        owner: dict[Value, str] = {}

        def augment(variable: str, visited: set[Value]) -> bool:
            for value in domains[variable]:
                if value in visited:
                    continue
                visited.add(value)
                if value not in owner or augment(owner[value], visited):
                    owner[value] = variable
                    return True
            return False

        for variable in sorted(variables, key=lambda name: len(domains[name])):
            if not augment(variable, set()):
                raise Wipeout(variable)
        return {variable: value for value, variable in owner.items()}

    def _reachable(self, edges: dict[Node, list[Node]], starts: list[Node]) -> set[Node]:
        reached = set(starts)
        queue = deque(starts)
        while queue:
            for successor in edges.get(queue.popleft(), ()):
                if successor not in reached:
                    reached.add(successor)
                    queue.append(successor)
        return reached

    def _components(self, edges: dict[Node, list[Node]], nodes: list[Node]) -> dict[Node, int]:
        """Tarjan's strongly connected components: nodes in the same component get the same number."""
        index: dict[Node, int] = {}
        low: dict[Node, int] = {}
        stack: list[Node] = []
        on_stack: set[Node] = set()
        component: dict[Node, int] = {}

        def connect(node: Node) -> None:
            index[node] = low[node] = len(index)
            stack.append(node)
            on_stack.add(node)
            for successor in edges.get(node, ()):
                if successor not in index:
                    connect(successor)
                    low[node] = min(low[node], low[successor])
                elif successor in on_stack:
                    low[node] = min(low[node], index[successor])
            if low[node] == index[node]:
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component[member] = index[node]
                    if member == node:
                        break

        for node in nodes:
            if node not in index:
                connect(node)
        return component
