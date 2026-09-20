import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace

from openmind.inference.service.covering_learner import CoveringLearner
from openmind.predictor.model.consequence import Consequence
from openmind.predictor.model.drawn import Always, Asked, Column, Drawn, More, Other, Row, Standing
from openmind.predictor.service.consequence_drawer import ConsequenceDrawer
from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.grid import Grid
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.change import Change, Moved, Placed, Removed, Told
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: One thing seen: the position, the action played in it, who played it, what it changed, and how it reads.
Seen = tuple[State, Action, Value, tuple[Change, ...], Mapping[str, Value]]


class ConsequenceLearner:
    """Works out what an action does, from having watched it done.

    What a game hands back is a position, and the changes between that and the one before are what the action did
    *this time*: a piece taken on b5, a square emptied at a7. Said that way nothing carries to the next position.
    So each part of each change is asked where it could have come from — is b5 the row of the source and the column
    of the target? is the queen put down the promotion that was asked for? — and the answers that hold every time
    are what the action does, said in terms of the action.

    A consequence that happens whenever the action is played needs nothing more. One that happens only sometimes
    needs to say when, and that is the same question as which actions are legal, asked of a different thing: the
    conditions are learned by covering, positive where the change happened and negative where the action was played
    and it did not."""

    def __init__(self, covering: CoveringLearner | None = None, drawer: ConsequenceDrawer | None = None) -> None:
        self._covering = covering or CoveringLearner()
        self._drawer = drawer or ConsequenceDrawer()

    def learn(
        self,
        seen: Sequence[Seen],
        players: Sequence[Value],
        within: Sequence[object] = (),
        positions: int = 2,
        grow: float = 0.67,
    ) -> tuple[Consequence, ...]:
        """What the actions in that evidence do, as consequences with the conditions under which they happen."""
        made: dict[Consequence, set[tuple[int, Change]]] = {}
        wanted: set[tuple[int, Change]] = set()
        places: dict[Consequence, set[object]] = {}
        for number, (state, action, acting, changes, _) in enumerate(seen):
            where = within[number] if len(within) == len(seen) else number
            for change in changes:
                wanted.add((number, change))
                for template in self._templates(change, state, action, acting, players):
                    made.setdefault(template, set()).add((number, change))
                    places.setdefault(template, set()).add(where)
        logger.info("Saw %d changes and %d ways of saying what made them", len(wanted), len(made))
        found: list[Consequence] = []
        left = set(wanted)
        while left:
            template = self._best(made, left, places, positions)
            if template is None:
                break
            explained = made[template] & left
            if self._always(template, seen, players):
                found.append(template)
            else:
                when = self._when(template, seen, {number for number, _ in made[template]}, players, within, positions, grow)
                if not when:
                    del made[template]
                    continue
                found.append(replace(template, when=when))
            left -= explained
            del made[template]
        logger.info("Said what an action does in %d consequences, %d changes unaccounted for", len(found), len(left))
        return tuple(found)

    def _best(
        self,
        made: Mapping[Consequence, set[tuple[int, "Change"]]],
        left: set[tuple[int, "Change"]],
        places: Mapping[Consequence, set[object]],
        positions: int,
    ) -> Consequence | None:
        """The description that accounts for most of what is left, the most general one where several tie.

        Every change can be described many ways, and all of them are true of the change they were read from. What
        tells a rule from a memory is how much else it accounts for: `the row of source and the column of target`
        explains a taking in every position, while `the row of source and 5` explains the one it was read from and
        nothing that is not on that file. So the description that explains the most is taken, what it explains is
        set aside, and the rest are asked again — a memorised description never comes up, because what it would
        have explained is already explained.

        A description seen in fewer positions than asked for is no description at all, however much it explains
        there: a rule of a game holds in more than one position."""
        best, worth = None, ()
        for template, explains in made.items():
            if len(places.get(template, ())) < positions:
                continue
            covers = len(explains & left)
            if not covers:
                continue
            held = (covers, -self._constants(template))
            if worth == () or held > worth:
                best, worth = template, held
        return best

    def _constants(self, template: Consequence) -> int:
        """How much of a description is a value written out rather than drawn from the action."""
        return sum(isinstance(one, Always) for one in (*template.where, *template.onto))

    def _always(self, template: Consequence, seen: Sequence[Seen], players: Sequence[Value]) -> bool:
        """Whether that consequence happens every time an action of its kind is played."""
        for state, action, acting, changes, _ in seen:
            made = self._drawer.drawn(template, state, action, acting, players)
            if made is not None and made not in changes:
                return False
        return True

    def _when(
        self,
        template: Consequence,
        seen: Sequence[Seen],
        happened: set[int],
        players: Sequence[Value],
        within: Sequence[object],
        positions: int,
        grow: float,
    ) -> tuple:
        """The conditions under which it happens, learned from the actions it happened to and those it did not."""
        examples, places = [], []
        for number, (state, action, acting, _, readings) in enumerate(seen):
            if self._drawer.drawn(template, state, action, acting, players) is None:
                continue
            examples.append((readings, number in happened))
            places.append(within[number] if len(within) == len(seen) else None)
        if not any(held for _, held in examples):
            return ()
        return self._covering.learn(
            examples, within=places if len(within) == len(seen) else (), positions=positions, grow=grow
        )

    def _templates(
        self, change: Change, state: State, action: Action, acting: Value, players: Sequence[Value]
    ) -> list[Consequence]:
        """Every way that change could have been drawn from its action."""
        if isinstance(change, Told):
            return [
                Consequence("Told", change.model, value=one)
                for one in self._values(change.value, state, action, acting, players)
            ]
        name = type(change).__name__
        if isinstance(change, Moved):
            wheres = self._squares(change.source, state, action)
            ontos = self._squares(change.target, state, action)
            return [Consequence(name, change.model, where, onto) for where in wheres for onto in ontos]
        if isinstance(change, Removed):
            return [Consequence(name, change.model, where) for where in self._squares(change.at, state, action)]
        return [
            Consequence(name, change.model, where, value=one)
            for where in self._squares(change.at, state, action)
            for one in self._values(change.value, state, action, acting, players)
        ]

    def _squares(self, at: Coordinates, state: State, action: Action) -> list[tuple[Drawn, ...]]:
        """Every way that square could have been named in terms of the action."""
        rows: list[Drawn] = []
        columns: list[Drawn] = []
        for name, value in action.parameters:
            cell = self._cell(state, value)
            if cell is None:
                continue
            if cell[0] == at[0]:
                rows.append(Row(name))
            if cell[1] == at[1]:
                columns.append(Column(name))
        rows.append(Always(at[0]))
        columns.append(Always(at[1]))
        return [(row, column) for row in rows for column in columns]

    def _values(
        self, value: Value, state: State, action: Action, acting: Value, players: Sequence[Value]
    ) -> list[Drawn]:
        """Every way that value could have been drawn."""
        found: list[Drawn] = []
        for name, held in action.parameters:
            if held == value:
                found.append(Asked(name))
            cell = self._cell(state, held)
            if cell is None:
                continue
            for model, grid in self._grids(state).items():
                if grid.inside(cell) and grid.at(cell) == value:
                    found.append(Standing(model, name))
        others = [player for player in players if player != acting]
        if len(others) == 1 and value == others[0]:
            found.append(Other())
        for name, model in state.models:
            held = getattr(model, "value", None)
            if isinstance(held, int | float) and not isinstance(held, bool) and isinstance(value, int | float):
                if not isinstance(value, bool) and value - held in (1, -1):
                    found.append(More(name, int(value - held)))
        found.append(Always(value))
        return found

    def _cell(self, state: State, value: Value) -> Coordinates | None:
        for grid in self._grids(state).values():
            if grid.aliases is None or not isinstance(value, str):
                continue
            try:
                return grid.aliases.to_coordinates(value)
            except (ValueError, KeyError):
                continue
        return None

    def _grids(self, state: State) -> dict[str, Grid]:
        return {name: model for name, model in state.models if isinstance(model, Grid)}
