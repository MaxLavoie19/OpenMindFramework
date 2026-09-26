from collections.abc import Mapping
from dataclasses import dataclass

from openmind.structure.model.domain import Numbers
from openmind.structure.model.grid_aliases import GridAliases
from openmind.structure.model.kind import Kind
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class GridKind:
    """A grid of a given extent whose cells are all of one kind."""

    holds: Kind
    shape: tuple[int, ...]
    names: GridAliases | None = None

    def cells(self) -> tuple[Value, ...]:
        """What a game calls each of its cells, where it names them — which is what a parameter pointing at a cell
        ranges over. Empty where the grid has no names, and then a game addresses cells some other way."""
        if self.names is None:
            return ()
        rows, columns = self.shape
        return tuple(self.names.to_alias((row, column)) for row in range(1, rows + 1) for column in range(1, columns + 1))

    def reaches(self, numbers: Numbers) -> Numbers:
        """Those numbers narrowed to the offsets that could carry something across this grid.

        **A window, and not a cap.** A game declares a move as whole numbers and says nothing about eight, so
        something must decide how far out to offer candidates or there is no listing them at all. Nothing outside
        the widest span of this grid can land on it from *any* cell, so nothing outside is refused for a reason
        the inside does not already show — the window loses nothing, and no number came from the game or was
        chosen here.

        **And the edge stays learnable, which is the point of checking.** Whether an offset leaves the grid
        depends on where the mover stands, not on the offset alone: from a corner every negative offset goes off
        it. So candidates that leave the grid are abundant inside this window, and *you may not move off it* is a
        rule with evidence rather than one assumed away by the listing.

        The widest span and not each axis's own, because nothing here knows which parameter is which axis. On a
        grid that is not square that offers a few offsets no cell could use, which are refused like any other."""
        widest = max(self.shape) - 1 if self.shape else 0
        return numbers.within(-widest, widest)


@dataclass(frozen=True, slots=True)
class ListKind:
    holds: Kind


@dataclass(frozen=True, slots=True)
class MapKind:
    keys: Kind
    holds: Kind


#: What a named model of a state is made of.
type ModelKind = Kind | GridKind | ListKind | MapKind


#: What one parameter of an action may be: something listed, or a set of numbers that need not be.
type ParameterKind = Kind | Numbers


@dataclass(frozen=True, slots=True)
class ActionKind:
    """An action and what each of its parameters may be. The parameters are the variables of the constraint
    problem, and their kinds are the domains those variables range over.

    **The thing that acts is a parameter like the rest.** A move by a thing standing somewhere is a move whose
    first parameter points at where it stands, constrained to one value — so which rules apply is settled inside
    the constraint problem rather than by something dispatching outside it. It points at the place and not at the
    thing, which is how what moves knows where it is without anything saying twice what the grid's own index
    already says."""

    name: str
    parameters: tuple[tuple[str, ParameterKind], ...]

    @property
    def domains(self) -> Mapping[str, tuple[Value, ...]]:
        """What each parameter ranges over as declared, which for an unbounded set of numbers is nothing.

        Empty means "not listable here", not "no values". `Schema.domains` is what resolves those, because what
        bounds them is the rest of the game rather than the action."""
        return {name: kind.domain for name, kind in self.parameters}


@dataclass(frozen=True, slots=True)
class Schema:
    """What a game's states are made of and what its actions may be.

    Declared by the game, once, and never inferred. It is what lets OMF lay out the whole space of candidates
    before it has seen a single move — the four thousand assignments two cell parameters allow — rather than
    working from whatever a game happened to show it.

    It says what things may be and never what they mean. That every square has a colour is here; that a bishop
    stays on one colour is not, and is the sort of thing there is to learn."""

    models: tuple[tuple[str, ModelKind], ...]
    actions: tuple[ActionKind, ...] = ()

    def model(self, name: str) -> ModelKind:
        for held, kind in self.models:
            if held == name:
                return kind
        raise KeyError(f"The schema has no model called {name!r}")

    def action(self, name: str) -> ActionKind:
        for held in self.actions:
            if held.name == name:
                return held
        raise KeyError(f"The schema has no action called {name!r}")

    def domains(self, action: str) -> Mapping[str, tuple[Value, ...]]:
        """What each parameter of that action ranges over: the whole domain, before any constraint narrows it.

        **A set of numbers the game left unbounded is bounded here, by the game's own grids.** A game says a move
        takes whole numbers and says nothing about how far, which is what leaves *you may not move off the grid*
        to be learned instead of handed over — but nothing can list an unbounded set, so the widest grid this
        game has says how far out candidates are worth offering. Nothing beyond it could land on that grid from
        any cell, so nothing beyond is refused for a reason the inside does not already show.

        **This is an assumption and worth naming as one.** It reads numbers in an action as offsets across the
        game's grid, because a game holding both and declaring one unbounded leaves no other way to enumerate. A
        game with no grid at all keeps its numbers unlistable, and whoever tries to enumerate them gets nothing
        rather than a number somebody invented."""
        found: dict[str, tuple[Value, ...]] = {}
        for name, kind in self.action(action).parameters:
            if isinstance(kind, Numbers) and not kind.listable:
                kind = self._reached(kind)
            found[name] = kind.domain
        return found

    def _reached(self, numbers: Numbers) -> Numbers:
        """Those numbers narrowed by the widest grid this game has, or left alone where it has none."""
        grids = [kind for _, kind in self.models if isinstance(kind, GridKind) and kind.shape]
        if not grids:
            return numbers
        return max(grids, key=lambda one: max(one.shape)).reaches(numbers)
