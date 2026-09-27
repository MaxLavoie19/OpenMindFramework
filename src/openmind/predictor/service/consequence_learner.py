import logging
from collections.abc import Callable, Sequence
from dataclasses import fields

from openmind.inference.model.example import Example
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.inference.service.refusal_learner import RefusalLearner
from openmind.statement.model.consequence import Consequence
from openmind.structure.model.map import Map
from openmind.statement.model.drawn import Always, Drawn, Other, Place, Stepped
from openmind.predictor.model.watched import Watched
from openmind.structure.model.record import Record
from openmind.structure.model.schema import ActionKind
from openmind.structure.model.value import Value
from openmind.world.model.state import State
from openmind.statement.model.change import Change

logger = logging.getLogger(__name__)

#: Where a change's parts live, by the kind of change it is.
PLACES = {"Placed": ("at",), "Removed": ("at",), "Moved": ("source", "target"), "Told": ()}


class ConsequenceLearner:
    """What an action does, learned from having watched it done.

    **Said as the doing, not as the position it left.** A change naming the square a pawn was taken on says what
    happened once; the same change with that square drawn from the action — the row of where the move started and
    the column of where it landed — says what happens whenever the action is played. So what is learned here is
    not which squares changed but *how the squares that changed are worked out from the action*.

    **That is what ties a move to the thing it moves.** Once it is known that a move carries whatever stands
    where the origin points, the piece doing the move is no longer one of a dozen things the position holds that
    a constraint might be about: it is the thing the change names. What the predictor concludes becomes what the
    constraint learner can say, which is the whole of why these two halves are one design.

    **Drawn by agreement, not by search.** Each sighting says which of the action's places hold the numbers the
    change used; what survives every sighting is the drawing. A square that matched the origin once and nothing
    the next time was a coincidence, and it falls out without anything having to score it.

    It keeps nothing: built once, it is given the sightings on every call."""

    def __init__(
        self,
        action: ActionKind | None = None,
        readings: CandidateReadings | None = None,
        learner: RefusalLearner | None = None,
        acting: Callable[[State], Value] | None = None,
        players: Sequence[Value] = (),
    ) -> None:
        self._action = action
        self._readings = readings
        self._learner = learner
        # Who acted, read from the position by whoever knows which of its structures says so, and who the
        # players are. Both only so that "the player whose action this is not" can be offered as a drawing:
        # without it, whose turn it becomes is unlearnable, since the value differs every sighting and no
        # constant survives them.
        self._acting = acting
        self._players = players

    def learn(self, watched: Sequence[Watched], starting: Sequence[Consequence] = ()) -> tuple[Consequence, ...]:
        """What those sightings say the action does.

        One consequence per kind of change per structure **per action**. A game of several phases has an
        action per phase and each does its own thing, so what is learned is what *this* action leads to and
        never what actions lead to. That is as far as this goes: an action that moves two
        things at once — a castling, a card dealt to each player — has two changes of one kind, and telling which
        sighting's first is which sighting's second is not something agreement alone can do. Those are left
        unlearned rather than learned wrongly, and saying which they are is the caller's answer."""
        found: dict[tuple[str, str, str], list[tuple[Watched, Change]]] = {}
        # Where each kind of change stood among an action's changes, as a share of the way through. The game
        # shows it in every sighting and it is not decoration: a removal before a move is a capture, and the same
        # two the other way round is a piece moving onto a square and then being deleted off it.
        #
        # **A share of the way through and not the place itself**, because a kind of change does not keep one
        # place. A move is the first change of a quiet move and the second of a capture, so the earliest place it
        # was ever seen in is nought — the same as the removal's — and the two sort as equals, which is the fault
        # this was written to fix. Where it stands *among however many there are* says what the places cannot.
        standing: dict[tuple[str, str, str], list[float]] = {}
        for one in watched:
            seen: dict[tuple[str, str, str], int] = {}
            for at, change in enumerate(one.changes):
                key = (one.action.name, type(change).__name__, self._about(one, change))
                seen[key] = seen.get(key, 0) + 1
                standing.setdefault(key, []).append(at / max(len(one.changes) - 1, 1))
                found.setdefault(key, []).append((one, change))
            for key, held in seen.items():
                if held > 1:
                    found.pop(key, None)
                    logger.debug("Left %s alone: it happens more than once in a sighting", key)
        ordered = {
            key: rank
            for rank, key in enumerate(
                sorted(standing, key=lambda one: sum(standing[one]) / len(standing[one]))
            )
        }
        learned = []
        for held, sightings in found.items():
            action, kind, named = held
            model = named.split(" at ")[0]
            drawn = self._drawn(kind, sightings)
            if drawn is None:
                continue
            where, onto, value, settled = drawn
            learned.append(
                Consequence(
                    kind, action, model, where, onto, value,
                    when=self._when(sightings, action, watched),
                    order=ordered[held],
                    settled=settled,
                )
            )
        logger.info("Learned %d consequences from %d sightings", len(learned), len(watched))
        return tuple(learned)

    def _about(self, watched: Watched, change: Change) -> str:
        """Which structure a change is of, and which entry of it where entries are named.

        **Two changes of a kind are left unlearned because nothing tells them apart, and a map's entries tell
        themselves apart.** The rule is right for a board: an action moving two pieces gives two moves of one
        grid, and which sighting's first goes with which sighting's second cannot be had from agreement. It is
        wrong for something held by key — what a game pays each player is one change per player, under that
        player's own name, and the names are the same in every sighting. Grouped by name they are as
        unambiguous as any single change, and a game being won became learnable the moment they were.

        A grid's coordinates are not used the same way, and must not be: they move with the action, so keying
        on them would make every sighting a group of its own and nothing would ever generalise."""
        held = watched.where.model(change.model) if watched.where.has(change.model) else None
        at = getattr(change, "at", None)
        if not isinstance(held, Map) or at is None:
            return change.model
        return f"{change.model} at {at[0] if len(at) == 1 else at}"

    def _when(self, sightings, action: str, watched: Sequence[Watched]) -> tuple:
        """The conditions under which that change happens, or none where it happens every time.

        **A consequence with no conditions is predicted always**, and that is right for most of a move and wrong
        for the rest: a capture does not happen every move, nor does a castling's rook, nor a promotion. Left
        empty, the predictor says every move takes something.

        Learned by the machinery that learns what a game refuses, pointed at a different question. The sightings
        where the change happened are what a clause must cover; the sightings of the same action where it did
        not are what it must not. Where there are none of the second, the change happens whenever the action is
        played and there is nothing to condition it on — which is the honest empty answer rather than an
        untested one."""
        if self._readings is None or self._learner is None:
            return ()
        happened = {id(one) for one, _ in sightings}
        cases = [
            Example(
                self._readings.read(one.where, one.action),
                id(one) in happened,
                one.where,
            )
            for one in watched
            if one.action.name == action
        ]
        if all(one.holds for one in cases) or not any(one.holds for one in cases):
            return ()
        return self._learner.learn(cases, InferenceBudget(seconds=5.0))

    def _drawn(
        self, kind: str, sightings: Sequence[tuple[Watched, Change]]
    ) -> tuple[tuple[Drawn, ...], tuple[Drawn, ...], Drawn | None, bool] | None:
        """How that change's parts are worked out from the action, or None where the sightings disagree.

        Also whether every place narrowed to one drawing. Where several still fit, one of them is shown — and
        saying which case that is, is the difference between thin evidence and a confident wrong answer."""
        places = PLACES.get(kind)
        if places is None:
            return None
        drawn: list[tuple[Drawn, ...]] = []
        narrowed = True
        for name in places:
            agreeing = self._agreeing(sightings, name)
            if agreeing is None:
                return None
            held, alone = agreeing
            narrowed = narrowed and alone
            drawn.append(held)
        value = self._value(sightings) if kind in ("Placed", "Told") else None
        return (
            drawn[0] if drawn else (),
            drawn[1] if len(drawn) > 1 else (),
            value,
            narrowed,
        )

    def _agreeing(
        self, sightings: Sequence[tuple[Watched, Change]], name: str
    ) -> tuple[tuple[Drawn, ...], bool] | None:
        """That part of the change, drawn the one way every sighting allows, and whether it is the only way.

        More than one drawing can survive every sighting so far — a square that is both the mover's column and
        the mover's row, while the two happen to agree. One is shown, and the caller is told it is one of
        several, because a reader cannot otherwise tell a settled drawing from a coin toss."""
        together: list[set[Drawn] | None] = []
        for watched, change in sightings:
            held = getattr(change, name, None)
            if held is None:
                return None
            numbers = held if isinstance(held, tuple) else (held,)
            if not together:
                # Nothing known yet, which is not the same as nothing possible. Started at the empty set and
                # then narrowed, the first sighting empties every place and nothing is ever drawn.
                together = [None for _ in numbers]
            if len(numbers) != len(together):
                return None
            for number, one in enumerate(numbers):
                ways = self._ways(watched, one)
                held_ways = together[number]
                together[number] = ways if held_ways is None else held_ways & ways
        if not together or any(not one for one in together):
            return None
        return (
            tuple(sorted(one, key=str)[0] for one in together if one),
            all(len(one) == 1 for one in together if one),
        )

    def _ways(self, watched: Watched, value: Value) -> set[Drawn]:
        """Every way that value could have been drawn from the action, before the sightings narrow it.

        A place a parameter points at, and that place stepped on by what another parameter says. The second is
        needed wherever a move names how far it goes rather than where it lands: the landing is then neither
        parameter but one plus the other, and without it the only agreed drawing is none, so nothing is said
        about the move at all.

        **And the player whose action this is not**, which is the one drawing here that comes from neither the
        action nor the position. Whose turn it becomes is black after a white move and white after a black one,
        so no constant survives the sightings and no place of the action holds it — the drawing agreed on was
        none, and the position a move led to came back with no turn in it at all. Every rule that reads whose
        turn it is then matches nothing, which does not fail: it quietly makes the other side's moves all legal.

        Still not offered, and so still unlearnable: a count that goes up (`More`), and what a model holds where
        a parameter points (`Standing`). Neither is needed for a move; a clock is drawn as nothing because of
        the first."""
        found: set[Drawn] = {Always(value)}
        if self._acting is not None and self._players:
            others = [one for one in self._players if one != self._acting(watched.where)]
            if len(others) == 1 and others[0] == value:
                found.add(Other())
        places = [
            (name, place, one)
            for name, held in watched.action.parameters
            for place, one in self._parts(name, held)
        ]
        for name, place, one in places:
            if one == value:
                found.add(Place(name, place))
            if not isinstance(one, (int, float)) or isinstance(one, bool):
                continue
            for other, _, step in places:
                if isinstance(step, (int, float)) and not isinstance(step, bool) and one + step == value:
                    found.add(Stepped(name, place, other))
        return found

    def _parts(self, parameter: str, value: Value) -> tuple[tuple[str, Value], ...]:
        """A parameter's places with what the game calls them.

        A parameter of several places is named by them; one of a single place is named by the kind the game
        declared, where that kind has a name to give. A set of numbers has none — it is not a thing a game named,
        it is a set OMF provides — so such a parameter is called what the action calls it, which is the honest
        answer and the one a rule reads anyway."""
        if isinstance(value, Record):
            return tuple((one.name, getattr(value, one.name)) for one in fields(value))
        declared = next((kind for name, kind in (self._action.parameters if self._action else ()) if name == parameter), None)
        return ((getattr(declared, "name", None) or parameter, value),)

    def _value(self, sightings: Sequence[tuple[Watched, Change]]) -> Drawn | None:
        """What the change puts down or a scalar now reads, drawn the one way every sighting allows."""
        together: set[Drawn] | None = None
        for watched, change in sightings:
            held = getattr(change, "value", None)
            ways = self._ways(watched, held)
            together = ways if together is None else together & ways
        return sorted(together, key=str)[0] if together else None
