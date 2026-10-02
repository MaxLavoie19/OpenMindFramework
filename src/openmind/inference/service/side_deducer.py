import collections
import logging
from collections.abc import Callable, Collection, Mapping, Sequence

from openmind.inference.model.fact import Fact
from openmind.inference.service.information import Information
from openmind.inference.model.sides import Sides
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number, Term
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: What is deduced, said as facts rather than held in a shape of this service's choosing.
#:
#: **A deduction is a thing learned and belongs where learned things go.** Whose a piece is and which way a
#: player faces were kept in a `Sides` record with a lookup on it, passed from service to service — which fixes
#: what may be owned (a bare value standing in a model, never a part of one) and fixes who may ask. Said as
#: facts, neither is fixed: what the deduction finds is what it says, and whatever needs it reads it.
OWNS = "owns"
FACES = "faces"


class SideDeducer:
    """Works out what belongs to whom, and which way each player faces, from the actions a game allows.

    A player's own side is the one thing about a board game that is nowhere on the board. It can be assumed — the
    first player faces up, the second faces down — but that is false of a game where both play the same way up, and
    an assumption that is false of some games is what OMF is not allowed to hold. It can be declared, but the whole
    exercise is that the integrator declared nothing.

    So it is read off what the game allows. A value is a player's where that player's actions move it: whatever
    stands where a move begins belongs to whoever may play it. And a player faces the way their one-way pieces go —
    a value of theirs that changes the row in one direction and never the other, in position after position, is a
    piece that advances, and a piece that advances says which way forward is. One such value orients the whole side;
    a game that has none has no side to tell, and then nothing is deduced and nothing is offered."""

    def __init__(self, information: Information | None = None) -> None:
        # What measures how much a reading says about whose a thing is. The same measure that tells a symbol
        # carrying its part from one that merely appears often, pointed at readings instead of symbols.
        self._information = Information() if information is None else information

    def deduce(
        self,
        examples: Sequence[tuple[Sequence[Literal], Value]],
        positions: Sequence[object] = (),
        least: int = 2,
        reaching: Callable[[], Mapping[tuple[Value, Value], Collection[Sequence[int]]]] | None = None,
    ) -> tuple[Literal, ...]:
        """What the game's allowed actions say about whose things are and which way each player faces, as facts.

        `examples` are the readings of actions the game allows, each with the player who may play it. `positions`
        says which position each was read in, and `least` how many positions a one-way value must be seen going one
        way in — seen in a single position, a rook that happened to move up twice is as one-way as a pawn.

        **Facts and not a record, because this is a thing learned.** What comes back is `owns(player, thing)` and
        `faces(player, step)` — for chess, that white moves up the board and black moves down, which is a sentence
        about the game rather than a field of a class. Nothing here settles what may own or be owned: a thing is
        whatever the readings said stood where a move began, so a game whose pieces are records, or colours, or
        numbers, says what it says. The old record could hold only a bare value standing in a model, and the one
        grid the constraint learner reads holds records, so it could say nothing at all about it.

        `reaching` is an optional second route to which way a player faces: what each thing of theirs could
        move by, from the rules rather than from what was seen. Given none, the facings are read off the
        examples as before, which is what a run that is still learning the rules has to do.

        **Both, because they fail differently.** Watching can only ever say a thing *has not yet* gone the
        other way: a rook kept on one file all game is one-way in the record, and no amount of watching
        settles it — the claim rests on an absence. The rules say what a thing *can* do, which is a presence
        and needs no sample. But a run early enough has no rules to ask, since the rules are what it is
        learning. So the derived answer is preferred where there is one, the watched answer stands where there
        is not, and the two agreeing is worth more than either — they are independent of each other, which is
        what makes their agreement evidence rather than arithmetic."""
        where = list(positions) if len(positions) == len(examples) else [None] * len(examples)
        owning, seen = self._owning(examples)
        facing = self._settled(self._facing(examples, owning, where, least), self._reached(reaching, owning))
        telling = self._telling(examples)
        standing = {
            (model, value, player): self._standing(seen.get((model, value), 0), telling.get(model, 0.0))
            for model, value, player in owning
        }
        owning = tuple(sorted(owning, key=lambda one: (-standing[one], repr(one))))
        self._said(owning, seen, facing, standing, telling)
        return (
            *(
                Literal(
                    OWNS,
                    (Constant(player), Constant(model), Constant(value), Number(standing[(model, value, player)])),
                )
                for model, value, player in owning
            ),
            *(Literal(FACES, (Constant(player), Constant(about), Number(step))) for player, about, step in facing),
        )

    def _standing(self, seen: int, telling: float) -> float:
        """How firmly a claim is held: how often it was seen, times how much its reading says whose a thing is.

        **Two different ways of being wrong, and one number could not have caught both.** Every claim used to
        be stated alike, and a run over chess made fifty of them a position: a pawn its player moved four
        hundred times, a queen of the other colour seen once because that player happened to be on the move,
        and light and dark squares given wholesale to one side. Counting sightings tells the first from the
        second and says nothing about the third, which is the thickest of the lot.

        **How often, by Laplace.** `(p + 1) / (p + 2)` — precision with a prior, so one sighting is worth 0.67
        and four hundred is worth 0.998. It ranks rather than cuts, which is what keeps a number nobody
        reasoned to out of it. Plain precision is what this used to be, and its own failing is named in the
        literature: one example is perfectly consistent.

        **How much the reading says, in bits.** A reading whose values divide the players carries information
        about whose a thing is; one giving both square colours to one player carries none, however often it was
        seen. The same measure that was repaired for the mirror of this fault, where a symbol saying its part
        perfectly was thrown away for being rare — here a symbol saying nothing was kept for being common."""
        laplace = (seen + 1.0) / (seen + 2.0)
        return laplace * telling

    def facts(self, said: Sequence[Literal]) -> tuple[Fact, ...]:
        """Those facts as the knowledge base takes them, so a deduction outlives the run that made it.

        **Both hold a number now, and whose a thing is was the one that needed it.** Which way a player faces
        is a step and always went in as one. Whose a thing is was said to have no quantity about it — what it
        says being the whole of what it is about — and that was true of the claim and false of the claiming: a
        pawn seen four hundred times and a queen seen once went in alike, as did light and dark squares given
        wholesale to one side. What is held is how firmly, so the knowledge base can tell them apart and
        whatever reads them can weigh rather than believe."""
        return tuple(
            Fact(one.predicate, tuple(self._held(term) for term in one.arguments[:-1]), one.arguments[-1].value)
            if isinstance(one.arguments[-1], Number)
            else Fact(one.predicate, tuple(self._held(term) for term in one.arguments))
            for one in said
        )

    def _held(self, term: Term) -> Value:
        """What a term stands for, whichever kind it is.

        **A thing of parts is kept whole.** A vocabulary that says a piece as a term of its colour and its kind
        has a term whose own name is just `piece`, so reading the name off it makes every piece on the board
        the same thing — and then nothing belongs to anybody, silently. The term itself is what it stands for
        in that case, which compares and hashes as the thing it is."""
        if isinstance(term, Number):
            return term.value
        name = getattr(term, "name", None)
        return term if getattr(term, "arguments", None) else name

    def sides(self, said: Sequence[Literal]) -> Sides:
        """Those facts as the record the stand-in engine takes.

        One presentation of what `deduce` found, for `ActionReadings`, which asks a record a
        question rather than reading a fact. It goes when they go.

        **It serves one vocabulary and says so.** A fact names the reading a thing was seen in — `at color
        source` — where the record wants the model it stood in. Recovering one from the other is only possible
        because the engine this feeds reads `{model} at {parameter}` and nothing else, so the model is the word
        after the predicate. Fed anything else it finds no owner, which is the honest failure: the record cannot
        hold what the fact says.

        **And the reading is chosen once for everybody, not once per player.** Every signed reading is tried, so
        evidence that does not tell them apart yields several per player — and a record holding one number each
        only means anything if those numbers are of the same thing. Choosing per player put white's step from
        one reading beside black's from another and had them both facing the same way, which is not a thing two
        players can do on one axis. The reading covering the most players is taken, and what is left over is
        said in the log rather than silently dropped."""
        owning = []
        for one in said:
            if one.predicate != OWNS:
                continue
            named = str(one.arguments[1].name).split()
            if len(named) < 2 or named[0] != "at":
                continue
            owning.append((named[1], one.arguments[2].name, one.arguments[0].name))
        by_reading: dict[str, dict[Value, int]] = {}
        for one in said:
            if one.predicate == FACES:
                by_reading.setdefault(str(one.arguments[1].name), {})[one.arguments[0].name] = int(
                    one.arguments[-1].value
                )
        if not by_reading:
            return Sides(tuple(sorted(owning, key=repr)), ())
        about = sorted(by_reading, key=lambda held: (-len(by_reading[held]), held))[0]
        held = by_reading[about]
        if len(by_reading) > 1:
            logger.info(
                "%d readings run one way for somebody; the record holds one, so %r is taken, covering %d of %d players",
                len(by_reading), about, len(held), len({player for one in by_reading.values() for player in one}),
            )
        return Sides(tuple(sorted(owning, key=repr)), tuple(sorted(held.items(), key=repr)))

    def _owning(
        self, examples: Sequence[tuple[Sequence[Literal], Value]]
    ) -> tuple[tuple[tuple[str, Value, Value], ...], dict[tuple[str, Value], int]]:
        """What belongs to whom: a value only ever moved by one player is that player's, and how often each was seen.

        A value several players move belongs to none of them — a neutral piece either side may push — and saying so
        is better than handing it to whoever moved it more often.

        **How often comes back with it, because "only one player moved it" is a weak thing to have found.** A
        value seen three times, all three by the same player, satisfies that exactly as well as a pawn seen four
        hundred times does, and the two are not the same claim at all: the first is most likely a player who
        happened to go first, and the second is what owning something looks like. Nothing here weighs them —
        a cut-off picked here would be a number nobody reasoned to — but what rests on three sightings can be
        told apart from what rests on four hundred by whatever does the weighing."""
        movers: dict[tuple[str, Value], set[Value]] = {}
        seen: dict[tuple[str, Value], int] = {}
        for readings, player in examples:
            for model, value in self._moved(readings):
                movers.setdefault((model, value), set()).add(player)
                seen[(model, value)] = seen.get((model, value), 0) + 1
        owning = tuple(
            sorted(
                ((model, value, next(iter(who))) for (model, value), who in movers.items() if len(who) == 1),
                key=repr,
            )
        )
        return owning, seen

    def _facing(
        self,
        examples: Sequence[tuple[Sequence[Literal], Value]],
        owning: tuple[tuple[str, Value, Value], ...],
        where: Sequence[object],
        least: int,
    ) -> tuple[tuple[Value, str, int], ...]:
        """Which way each player faces: the way a thing of theirs goes, where it never goes the other way.

        What advances is a white pawn, and neither half of that says so: pawns are moved by both players, and the
        white things on the board include a rook that goes both ways. So a one-way thing is looked for in what
        stands where the move begins taken whole, and it is that player's where no other player ever moves it.

        **Which reading runs that way is part of the answer now, not something assumed.** It used to take the
        row step because the vocabulary had exactly one signed reading and everyone knew which. Every signed
        reading is tried instead, and the one that is really one-way for a player comes back named — so a game
        whose forward runs along something other than rows says so, and nothing here had to be told.

        **Only a reading that goes both ways somewhere can say which way anybody faces.** How far apart two
        places are, how many steps a move takes, how many things stand between: all of them are magnitudes and
        none is ever negative, so each is one-way for every player by construction and says nothing whatever
        about sides. Measured against chess, taking them at face value had both players facing the same way — a
        thing two players cannot do on one axis. A reading earns its place here by being seen negative for
        somebody and positive for somebody, which is what makes "forward" mean anything at all."""
        ways: dict[tuple[Value, tuple, str], set[int]] = {}
        seen: dict[tuple[Value, tuple, str], set[object]] = {}
        movers: dict[tuple, set[Value]] = {}
        # Which signs each reading was ever seen taking, over everybody — a reading never seen negative is
        # a magnitude and carries no direction, however one-way it looks for any one player.
        signed: dict[str, set[int]] = {}
        for (readings, player), place in zip(examples, where, strict=True):
            moved = self._moved(readings)
            if not moved:
                continue
            # **Each thing on its own as well as all of them together, and the cases settle which says it.**
            # Taken only together, a holding is everything standing at every place the action points at — the
            # thing that moved, the square's colour, whatever sits where it lands — and such a collection
            # almost never turns up twice, so nothing is ever seen in the two positions it takes to tell a
            # pawn from a rook that happened to go the same way twice. Measured: 432 allowed cases over
            # sixteen positions, every one of them a holding of its own, and no player facing anywhere.
            #
            # Taken only one at a time, a vocabulary that says what a thing is apart from whose it is would
            # offer `pawn`, which both players move, and `white`, which includes a rook that goes both ways —
            # the reason this took them whole to begin with. So both are offered, and the gates below keep
            # whichever is really one-way for one player. `lands on` already does this with pairings it cannot
            # tell apart.
            whole = tuple(sorted(moved, key=repr))
            for holding in ({whole} | {(one,) for one in moved}):
                movers.setdefault(holding, set()).add(player)
                for about, amount in self._stepping(readings):
                    ways.setdefault((player, holding, about), set()).add((amount > 0) - (amount < 0))
                    seen.setdefault((player, holding, about), set()).add(place)
            for about, amount in self._stepping(readings):
                signed.setdefault(about, set()).add((amount > 0) - (amount < 0))
        facing: dict[tuple[Value, str], set[int]] = {}
        for (player, holding, about), steps in ways.items():
            if len(signed.get(about, set())) < 2:
                continue
            if len(steps) != 1 or len(seen[(player, holding, about)]) < least or movers[holding] != {player}:
                continue
            facing.setdefault((player, about), set()).update(steps)
        found = tuple(
            sorted(((player, about, steps.pop()) for (player, about), steps in facing.items() if len(steps) == 1), key=repr)
        )
        for player, about, step in found:
            logger.info("%r only ever has %s going %s, so that is forward for them", player, about, step)
        return found

    def _telling(self, examples: Sequence[tuple[Sequence[Literal], Value]]) -> dict[str, float]:
        """How much each reading says about whose a thing is, in bits: knowing what stands there, how much
        better is the player known?

        **A reading that divides the players is a reading about ownership; one that does not is not.** Chess's
        pieces divide them perfectly — a black rook is only ever moved by black — and its square colours divide
        them not at all, since both players play over light and dark alike. Counted, the first carries a bit
        and the second carries none, and the difference is what tells an ownership reading from a reading that
        merely happened to be seen.

        Averaged over the reading's values by how often each was seen, so a reading that is telling about one
        rare value and silent about everything else is not mistaken for one that is telling throughout."""
        players: dict[str, collections.Counter] = {}
        by_value: dict[tuple[str, Value], collections.Counter] = {}
        for readings, player in examples:
            for model, value in self._moved(readings):
                players.setdefault(model, collections.Counter())[player] += 1
                by_value.setdefault((model, value), collections.Counter())[player] += 1
        found: dict[str, float] = {}
        for model, whole in players.items():
            seen = sum(whole.values())
            if not seen:
                continue
            bits = 0.0
            for (held, _), where in by_value.items():
                if held != model:
                    continue
                bits += sum(where.values()) / seen * self._information.told(whole, where)
            found[model] = max(bits, 0.0)
        return found

    def _said(
        self,
        owning: tuple[tuple[str, Value, Value], ...],
        seen: dict[tuple[str, Value], int],
        facing: tuple[tuple[Value, str, int], ...],
        standing: dict[tuple[str, Value, Value], float],
        telling: dict[str, float],
    ) -> None:
        """Everything deduced, named, with what each one rests on.

        **A count cannot be argued with.** This used to say `Deduced 54 values as a player's own` and stop there.
        Chess has twelve kinds of piece, so fifty-four is plainly too many, and the log gave nobody a way to see
        what the extra ones were. They turned up by accident, in another service's output, as a candidate asking
        whether a *square's colour* was the acting player's — which it cannot be, since both players play over
        the same light and dark squares. Named against the game, that is obviously false, and letting somebody
        find it obviously false is the whole of what this line is for.

        **How firm each one is, because most of them will not be firm.** These are hypotheses and they are not
        of a kind: a value the acting player moved four hundred times is owned about as clearly as anything here
        gets said, and a value moved three times by whoever happened to be on the move is an accident wearing
        the same words. Both arrive from `_owning` as `owns(player, thing)` with nothing to tell them apart, so
        what each rests on is said beside it, firmest first — and how firmly, which is the number a reader
        needs and the one that used to be missing entirely."""
        logger.info("Deduced %d values as a player's own, and which way %d players face", len(owning), len(facing))
        for model in sorted(telling, key=repr):
            logger.info("What %r reads says %.2f bits about whose a thing is", model, telling[model])
        by_player: dict[Value, dict[str, list[tuple[Value, int, float]]]] = {}
        for model, value, player in owning:
            firmness = standing.get((model, value, player), 0.0)
            by_player.setdefault(player, {}).setdefault(model, []).append((value, seen.get((model, value), 0), firmness))
        for player in sorted(by_player, key=repr):
            for model in sorted(by_player[player]):
                held = sorted(by_player[player][model], key=lambda one: (-one[2], repr(one[0])))
                logger.info(
                    "%s owns, of what %r reads: %s",
                    player,
                    model,
                    ", ".join(f"{self._named(value)} at {firm:.2f} on {times}" for value, times, firm in held),
                )

    def _named(self, value: Value) -> str:
        """A deduced thing as a name, whether it is one value or a thing of parts."""
        arguments = getattr(value, "arguments", None)
        if arguments is None:
            return "nothing" if value is None else str(value)
        return f"{getattr(value, 'name', '?')}({', '.join(self._named(self._held(one)) for one in arguments)})"

    def _moved(self, readings: Sequence[Literal]) -> list[tuple[str, Value]]:
        """Every thing the action's readings say stands where one of its parameters points.

        **Nothing here knows which reading means that, and it does not guess.** It used to: it found the
        reading spelled `rows from {first} to {second}`, took `first` as where the move began, and then took
        every reading whose name ended `" at first"`. Three pieces of string surgery, each true only of one
        vocabulary's spelling.

        **A parameter gives itself away.** An action reads each of its parameters under its own name, and the
        readings *about* a parameter name it among their terms — `source` is both `source = b1` and the middle
        term of `at(piece, source, pawn)`. Nothing else in a position does both: whose turn it is is read under
        its own name and no reading mentions it, and a model like `piece` is named by readings without ever
        being read on its own. So which terms are parameters falls out of the readings, and none of it depends
        on how any of them is spelled.

        **What stands there, and not the parameter itself.** `source = b1` is thing-valued and names a
        parameter, and it is the square rather than what is on it — taken as a thing it is different for every
        action and makes this a fingerprint of the move instead of a description of what it moves. A reading
        whose predicate is a parameter is that parameter, so it goes.

        **Both ends are offered and the evidence keeps one.** What stands where a move lands is as much a thing
        standing where a parameter points as what stands where it starts, and nothing here says which end a
        move begins at. It does not have to: both players land on the same squares, so what stands at the far
        end is moved by both and is owned by neither, and it falls out of `_owning` without being excluded
        here. That is what `lands on` already does with the pairings it cannot tell apart — offer both, and let
        the cases settle it."""
        named = self._parameters(readings)
        return [
            (self._about(one), self._held(one.arguments[-1]))
            for one in readings
            if one.arguments
            and one.predicate not in named
            and any(self._held(term) in named for term in one.arguments[:-1])
            and self._amount(one.arguments[-1]) is None
            and self._held(one.arguments[-1]) is not None
        ]

    def _parameters(self, readings: Sequence[Literal]) -> set[Value]:
        """Which of the action's readings are its parameters: read under their own name, and named by others.

        **However many numbers it takes to say where, and however deeply another reading names it.** Both
        halves of that were too narrow, and together they made this find nothing at all in chess: a square on
        a two-dimensional board is read `self(7, 1)` and not under one argument, and the reading that says
        what stands there says `holds(place(self, 1), pawn)` — naming `self` inside a term rather than as one.

        Measured before this: over 685 allowed cases, no parameter was found, so `_moved` gave nothing for
        every case, so every example was skipped before any gate, so no player ever faced anywhere and the
        whole orientation half of this service was dead. The ownership half runs by another path, which is why
        it looked half alive rather than broken.

        The mechanism was right and its reach was wrong. Nothing here is told how a square is spelled."""
        alone = {one.predicate for one in readings if one.arguments}
        named = {held for one in readings for term in one.arguments[:-1] for held in self._naming(term)}
        return {one for one in alone if one in named}

    def _naming(self, term: Term) -> set[Value]:
        """What a term names: what it stands for, and what the terms inside it name, however deep.

        A reading names a parameter whether it says it outright or carries it in a term of its own — `place`
        of a square and a coordinate is still a reading about that square."""
        found = {self._held(term)}
        for inside in getattr(term, "arguments", ()) or ():
            found |= self._naming(inside)
        return found

    def _stepping(self, readings: Sequence[Literal]) -> list[tuple[str, int]]:
        """Every signed number the action's readings carry, named by the reading that carried it.

        One of them is how far the move goes along the axis a player faces, and nothing says which. Offered all
        together, the one that is really one-way for a player is the one that survives position after position."""
        found = []
        for one in readings:
            if not one.arguments:
                continue
            amount = self._amount(one.arguments[-1])
            if amount is not None and amount != 0:
                found.append((self._about(one), amount))
        return found

    def _about(self, literal: Literal) -> str:
        """What a reading is about, as a name: its predicate and everything it named before what it read."""
        named = " ".join(str(self._held(one)) for one in literal.arguments[:-1])
        return f"{literal.predicate} {named}".rstrip()

    def _amount(self, term: Term) -> int | None:
        """That term as a whole number, or nothing where it is not one.

        Asked of both kinds, because the two vocabularies do not agree yet: one says a coordinate with `Number`
        and the other wraps every value in `Constant`, so a number arrives either way."""
        held = self._held(term)
        if isinstance(held, bool) or not isinstance(held, int):
            return None
        return held

    def _whose(self, owning: tuple[tuple[str, Value, Value], ...], model: str, value: Value) -> Value | None:
        for held, one, player in owning:
            if held == model and one == value:
                return player
        return None

    def _reached(
        self,
        reaching: Callable[[], Mapping[tuple[Value, Value], Collection[Sequence[int]]]] | None,
        owning: tuple[tuple[str, Value, Value], ...],
    ) -> tuple[tuple[Value, str, int], ...]:
        """Which way each player faces, read off what their things can do rather than what they were seen doing.

        **A thing is one-way when its reach is lopsided, and that is a fact about the rules.** Of everything on
        a chess board only a pawn can never go back, and it does not take a sample to know it: the places a
        pawn may reach are all on one side of it, and the places a rook may reach are on both. So the axis some
        thing is lopsided on is the axis of play, and the side it is lopsided to is that player's forward.

        Nothing here is told what a pawn is or which axis a board runs on. It is handed, for each thing and
        whose it is, the steps that thing could move by, and looks for an axis where every step has one sign.

        A thing that cannot move at all says nothing, and a game where everything can go both ways has no side
        to tell — which is true of draughts kings, and of noughts and crosses, and is the right answer there."""
        if reaching is None:
            return ()
        owners = {value: player for player, _, value in owning}
        facing: dict[tuple[Value, str], set[int]] = {}
        for (player, thing), steps in reaching().items():
            held = player if player is not None else owners.get(thing)
            if held is None or not steps:
                continue
            for axis in range(min(len(one) for one in steps)):
                signs = {(one[axis] > 0) - (one[axis] < 0) for one in steps} - {0}
                if len(signs) == 1:
                    facing.setdefault((held, f"axis {axis}"), set()).update(signs)
        found = tuple(
            sorted(
                ((player, about, steps.pop()) for (player, about), steps in facing.items() if len(steps) == 1),
                key=repr,
            )
        )
        for player, about, step in found:
            logger.info("%r has a thing that can only ever go %s along %s, so that is forward for them", player, step, about)
        return found

    def _settled(
        self,
        watched: tuple[tuple[Value, str, int], ...],
        derived: tuple[tuple[Value, str, int], ...],
    ) -> tuple[tuple[Value, str, int], ...]:
        """One answer out of the two routes, saying so where they differ.

        **The rules win where there are rules**, because watching rests on an absence — a rook that has not yet
        gone the other way is not a rook that cannot — while the rules say what a thing is able to do. Where
        there are none, what was watched stands, since a run still learning the rules has nothing else.

        **And where both speak, their agreeing is the thing worth having.** They share no machinery: one reads
        what was played, the other what is permitted. Two independent routes to one fact corroborate it in a
        way that two readings of the same move list never could."""
        if not derived:
            return watched
        if not watched:
            return derived
        agreed = sorted(set(watched) & set(derived), key=repr)
        apart = sorted(set(watched) ^ set(derived), key=repr)
        if agreed:
            logger.info("Watching the game and reading its rules agree on %d facings: %s", len(agreed), agreed)
        if apart:
            logger.info(
                "They part on %d, and the rules are taken: watching can only say a thing has not gone the "
                "other way yet. %s", len(apart), apart,
            )
        return derived
