"""The marker wall as a constraint problem, to put OMF's solver under a load no game gives it.

Colored markers are stapled along a wall, one column of them per metre, and the colors of the few columns in view
say which stretch of wall is being looked at. A window of `m` markers can show at most `colors ** m` patterns, and
that bound is reached exactly: what is wanted is a sequence in which every pattern occurs once.

Nodes are the window patterns, and there is an edge from one to another when the first's last columns are the
second's first columns, so the sequence is a cycle through every node — which is `circuit`, and nothing else.
"""

from collections.abc import Mapping, Sequence

from openmind.rule.model.domain_rule import DomainRule
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule

COLORS = 7
PAINT = "paint"
SUCCESSOR = "successor_{node}"


def alphabet(rows: int, colors: int = COLORS) -> int:
    """How many patterns a column of that many markers can show."""
    return colors**rows


def positions(rows: int, window: int, colors: int = COLORS) -> int:
    """How many stretches of wall a window that wide can tell apart, which is how many nodes there are."""
    return alphabet(rows, colors) ** window


def successors(node: int, rows: int, window: int, colors: int = COLORS) -> range:
    """The nodes that can follow this one: shift its first column off the front and put any column on the back."""
    symbols = alphabet(rows, colors)
    carried = (node % symbols ** (window - 1)) * symbols
    return range(carried, carried + symbols)


def encoding(rows: int, window: int, colors: int = COLORS) -> tuple[tuple[str, ...], dict[str, Rule], tuple[Rule, ...]]:
    """The parameters, their values rules and the one constraint.

    The values are kept as values rather than as source, which is what `DomainRule` is for: a hundred thousand
    parameters written as source would be a hundred thousand separate compiles of a hundred thousand rules."""
    names = tuple(SUCCESSOR.format(node=node) for node in range(positions(rows, window, colors)))
    values: dict[str, Rule] = {
        names[node]: DomainRule(tuple(successors(node, rows, window, colors))) for node in range(len(names))
    }
    return names, values, (PythonRule(f"circuit({', '.join(names)})"),)


def sequence(found: Mapping[str, int], rows: int, window: int, colors: int = COLORS) -> list[int]:
    """The column patterns along the wall, read off the cycle: walk it from node nought taking each node's first
    column, then repeat the first columns so the wall has two ends rather than joining up."""
    symbols = alphabet(rows, colors)
    total = positions(rows, window, colors)
    walk, node = [], 0
    for _ in range(total):
        walk.append(node)
        node = found[SUCCESSOR.format(node=node)]
    if node != 0:
        raise AssertionError("the walk does not come back to where it started")
    columns = [step // symbols ** (window - 1) for step in walk]
    return columns + columns[: window - 1]


def markers(column: int, rows: int, colors: int = COLORS) -> tuple[int, ...]:
    """The color of each marker of a column, the top one first."""
    return tuple((column // colors**row) % colors for row in range(rows))


def windows(columns: Sequence[int], window: int) -> list[tuple[int, ...]]:
    """Every stretch of wall the window can be held against, as the column patterns it shows."""
    return [tuple(columns[place : place + window]) for place in range(len(columns) - window + 1)]


def every_pattern(rows: int, window: int, colors: int = COLORS) -> set[tuple[int, ...]]:
    """Every window of columns there is, which is what the wall has to contain exactly once each."""
    symbols = alphabet(rows, colors)
    return {
        tuple((pattern // symbols**place) % symbols for place in reversed(range(window)))
        for pattern in range(positions(rows, window, colors))
    }
