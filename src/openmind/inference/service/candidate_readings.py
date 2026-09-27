import logging
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations, permutations, product

from openmind.inference.model.evidence import Evidence
from openmind.inference.model.example import Example
from openmind.inference.service.side_deducer import FACES, OWNS
from openmind.statement.model.literal import Literal
from openmind.statement.model.drawn import part, place as placed
from openmind.statement.model.term import Constant, Functor, Number, Term
from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.grid import Grid
from openmind.structure.model.list import List
from openmind.structure.model.machine import Machine
from openmind.structure.model.map import Map
from openmind.structure.model.record import Record
from openmind.structure.model.schema import ActionKind
from openmind.structure.model.scalar import Scalar
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: How two of the numbers a candidate's parameters hold stand to each other: which two places, which way, and
#: how far.
#:
#: Named apart from the arithmetic `apart`, which it is not. That one is three numbers and a question about
#: them; this is a reading of a case, saying which two places of which two parameters it is about and what
#: their distance came to. A clause grown from cases can only ever mention what a case said, so a distance
#: has to be said before any rule about travelling can exist.
PLACES_APART = "places apart"

#: How far a place reaches, whichever way it goes: the size of a step and not the step.
#:
#: A term of the place rather than a name built out of it. "how far self row" was a string spelling out
#: `how far(place(self, row))` and unable to be taken apart, so nothing could ask which place it was about —
#: which is exactly what pairing two of them by sort has to ask.
HOW_FAR = "how far"

#: What a structure holds where one of the candidate's parameters points: which parameter, which structure, and
#: the thing itself.
#:
#: A move is about the piece on the square it starts from, and until now nothing said so. It was *sayable* — the
#: parameter's places and the structure's places tied by two variables say it exactly — but a constraint is grown
#: from the readings a case carried, and finding that tie means picking one grid reading out of sixty-four by
#: pairing. Every rule about how a thing moves needs it, and every one of them had to find it again.
#:
#: Said outright it is one reading, and what a constraint has to discover is what *kind* of thing is there rather
#: than which of sixty-four squares to look at. Nothing about it is a board: it is what any structure holds where
#: any parameter points, for a game whose parameters point into structures at all.
POINTED_AT = "holds"

#: How far one of a candidate's numbers is from nothing, and which way: which place, which way, how far.
#:
#: **Two numbers being the same size is not two numbers being apart by nothing.** A thing that moves along a
#: diagonal moves by as much one way as the other, and there are four such diagonals: `places apart` between the
#: two offsets is nothing for two of them and twice the distance for the other two, so the one rule cannot be
#: said and two of the four directions have no rule at all. Said from nothing, both offsets carry the same
#: distance whichever way they point, and a clause holding one variable in both places says the whole of it.
#:
#: Nothing about this is a board or a diagonal. It is how large a number is apart from which side of nothing it
#: falls on, which is a question about signed numbers — a bet against a pot, a clock running back. Where a
#: parameter never goes below nothing it says again what the parameter's own reading said, and distilling takes
#: it out.
FROM_NOTHING = "from nothing"

#: What a structure holds where the candidate's numbers *reach*, some of them taken as a step from another:
#: which parameter it steps from, which places it steps by, which structure, and the thing found there.
#:
#: **Without it, a move said as a step has nowhere it lands.** `holds` reads what stands where a parameter
#: points, which was enough while a move named the square it went to. A move said as how far it goes names its
#: mover and its step, and the square it reaches is neither — it is their sum, and nothing here does arithmetic,
#: so the most basic rule there is, that you may not land on your own thing, would be out of reach.
#:
#: **Which step goes with which place is not declared, and both ways are offered.** Nothing knows that the first
#: number steps along rows; the pairing that is real earns its place in the rules and the other never does, so
#: the cases decide rather than a convention nobody wrote down.
#:
#: Where the sum falls outside the structure the reading still stands and says so, because *you may not move off
#: it* is a rule like the others and a rule cannot be built from a reading that is simply missing.
LANDS_ON = "lands on"

#: What a grid holds beyond its own edge, which is not what it holds in an empty cell.
OUTSIDE = "outside"

#: What stands on a cell lying on the line between where a candidate's places are and where its steps reach:
#: which parameter it steps from, which places it steps by, which structure, and the thing found there.
#:
#: **One reading per cell in between, because a refusal only has to find one.** "The way is clear" is a
#: statement about every cell between two others, which a clause with one conclusion cannot make; "something is
#: in the way" is a statement that *some* cell between them holds a thing, and that is what a refusal needs.
#: Writing the rules as refusals is what makes this sayable at all.
#:
#: **A line and not a ray.** A ray is cast from somewhere in a direction and presumes something travelling; a
#: line is geometry, and two cells either lie on one or they do not. It is the same relation a row of three
#: wants, or four in a row, or a chain of stones — none of which involves anything moving. Only cells strictly
#: between are said: where a thing lands is `lands on`, and where it stands is `holds`.
ON_LINE = "on line"

#: Which places of a reading hold a *value* rather than name what the reading is about, for the readings where
#: that is not simply the last one.
#:
#: **A reading names what it is about and then says what was read of it**, and generalising is held to the
#: difference: two readings naming different places are not two accounts of one thing, so their naming places
#: must agree. That is right for a place and a model, and wrong for these two — which way one number lies from
#: another is a *value*, as much as what stands on a square is.
#:
#: Left un-said, the way could never become a variable, so a thing that goes one way and a thing that goes the
#: other were two rules that nothing could join. Every pawn rule is written twice for that reason, and a
#: diagonal needs two rules where it should need one. It also contradicted what `_way` is for: said as a value
#: it can be tied to whoever is acting, and which side goes which way becomes a thing to be found.
VALUING: Mapping[str, tuple[int, ...]] = {PLACES_APART: (2,), FROM_NOTHING: (1,)}

#: Whose the thing standing where one of the candidate's parameters points is, said as a role: the one acting's,
#: another player's, or nobody's.
#:
#: **A rule about a thing of one's own cannot be said about a colour.** "You may not move another player's
#: thing" is one rule, and written over colours it is two that each name a side and neither of which says what
#: it means — and the rules that are about a player's *own* end of the board cannot be said at all. Whose a
#: thing is was deduced from the game's own actions before any of this; here it is read of a candidate so that
#: a constraint may be grown from it.
#:
#: Nothing about it is a board or a colour. It is who a thing belongs to, for a game that has players and things
#: at all, and a game where nothing belongs to anybody never produces it.
BELONGS = "belongs"

#: Who a thing belongs to, said as a role rather than as a player by name, so that one rule serves every side.
ACTING, ANOTHER = "the one acting", "another player"

#: How far one of the candidate's numbers goes the way the acting player faces.
#:
#: **Which way is forward is not a fact about the board.** A pawn's step is one row up for one player and one
#: row down for the other, so every pawn rule is learned twice and carries a colour inside a rule that is not
#: about colour. Read as how far it goes *toward where its own player faces*, both are one rule.
#:
#: **Offered for every number and not for the one that is really the axis**, because nothing here knows which
#: number is which — the same reason `lands on` offers both pairings of steps to places. A number that is not
#: the axis reads as itself times one or minus one, which discriminates nothing and is distilled away; the one
#: that is earns its place in the rules.
TOWARD = "toward"

#: The reading families the candidate itself is in, as against what a position says of places the candidate
#: never mentions.
TIED = (PLACES_APART, POINTED_AT, FROM_NOTHING, LANDS_ON, BELONGS, TOWARD)


class CandidateReadings:
    """Candidates in a position, said as ground literals.

    **This is the only place a reading can enter the engine**, so what it offers is the whole of what any rule OMF
    learns will ever be able to say. It is small for that reason and no other.

    What it offers is the state's own data models opened up — what stands in each cell of each grid, what each
    scalar says, what each map and list hold — together with where the candidate's parameters point. Nothing else.
    A grid can also give its diagonals, its lines, its rays, its neighbours and the distance between two cells, and
    none of those are offered, because they are the relations the learner is supposed to discover. Handing them
    over would be handing it the answer to a game we are pretending not to know.

    **The whole structure is read, not the part a candidate names.** Reading only the cells a move names is
    cheaper and cannot work: nothing in those two readings could ever tell a blocked way from a clear one, so the
    rule about sliding is not hard to find there, it is absent. Choosing instead to read "the cells between" would
    smuggle in a notion no game gave us. So everything is read and the learner is left to find what matters, and
    the cost of that is compute.

    **Coordinates are numbers, values are not.** A cell's row and column are read as numbers, so arithmetic and
    the comparisons apply to them and the learner can tie one to another; what stands in the cell is read as a
    thing, even where a game spells it with a digit. That distinction is the whole of what is assumed about a
    grid: that its cells are addressed by numbers. Which relations over those numbers mean anything is not
    assumed, and is what there is to learn.

    It keeps nothing but what the game declared: built once with the action's shape, it is given the position and
    the candidates on every call."""

    def __init__(
        self,
        action: ActionKind | None = None,
        sides: Sequence[Literal] | Callable[[], Sequence[Literal]] = (),
        acting: Callable[[State], Value] | None = None,
        paid: Sequence[str] = (),
    ) -> None:
        self._places = self._named(action)
        # Which of a position's models hold what the game has paid, left unread.
        #
        # **Legality is what produces a result, so a rule conditioned on the result is backwards.** A position
        # carrying what it came to is what lets a predictor see a game being won, and it has no business in a
        # rule about what may be played: a game is not over because somebody won, somebody won because the
        # game was over. Worse, where a game pays out precisely when nothing was legal, a constraint reading
        # the payoff restates the observation — correct, cheap, and teaching nothing about how anything moves.
        #
        # Named by the caller rather than recognised here, because what a game calls the thing it pays with is
        # the game's business; nothing in this service knows what a payoff is.
        self._paid = frozenset(paid)
        # What was deduced about whose things are and which way each player faces, as facts, and a way to read
        # whose action this is off a position. Given neither, nothing about sides is offered and every rule
        # about a thing of one's own stays a rule about a colour, which is what it was.
        #
        # **Asked for rather than held, where the caller gives a way of asking.** What the game allows is what
        # the sides are deduced from, so on the first position there is nothing to deduce them from yet and
        # they get better as the evidence does. Built once with the facts of the moment, a learner would carry
        # the facts of the moment before it had seen anything — which is none — for the whole run. It is the
        # same reason the hypothetical asks for the predictor's conclusions instead of keeping them.
        self._sides = sides
        self._acting = acting

    @property
    def sides(self) -> tuple[Literal, ...]:
        """What is known about whose things are and which way each player faces, as of now."""
        return tuple(self._sides() if callable(self._sides) else self._sides)

    def _named(self, action: ActionKind | None) -> tuple[tuple[str, int, str, str], ...]:
        """Each of the action's places: which parameter, which position in it, what the game calls it, and what
        kind it is.

        Taken from the schema and never guessed. Where a game declares none, the places are numbered and nothing
        is known to be of a different kind from anything else, which is the old behaviour."""
        if action is None:
            return ()
        found = []
        for name, kind in sorted(action.parameters, key=lambda held: held[0]):
            # A set of numbers has no parts and is one place of its own, named after the parameter. What it is
            # called as a kind is what it is: two parameters of numbers are alike, and a row and a column are
            # not, which is the whole of what this is for.
            parts = getattr(kind, "parts", ()) or ((name, kind),)
            found.extend(
                (name, number, part, getattr(held, "name", "") or kind.readable)
                for number, (part, held) in enumerate(parts, start=1)
            )
        return tuple(found)

    def cases(self, evidence: Evidence, domains: Mapping[str, Sequence[Value]]) -> tuple[Example, ...]:
        """Every assignment the domains allow, read, and marked by whether the game refuses it.

        A case **holds** where the game does not list the action. Covering a case is therefore refusing it, and
        being legal is nothing but no clause covering the case — which is what having no generators means in
        practice.

        Nothing is sampled and nothing is left out. The cases are the whole space the solver will later search, so
        a constraint that accounts for all of them here accounts for everything the solver can propose there."""
        readings = self.of_state(evidence.where)
        aliases = self._aliases(evidence.where)
        found = tuple(
            Example(
                (*readings, *self._of_parameters(one.parameters, aliases, evidence.where)),
                not evidence.allows(one),
                evidence.where,
            )
            for one in self.candidates(evidence, domains)
        )
        refused = sum(1 for one in found if one.holds)
        logger.info(
            "Read %d candidates of %s over %s, %d of them refused and %d allowed, each said in %d readings",
            len(found), evidence.action, ", ".join(sorted(domains)), refused, len(found) - refused,
            len(found[0].literals) if found else 0,
        )
        self._unforeseen(evidence, domains)
        return found

    def candidates(self, evidence: Evidence, domains: Mapping[str, Sequence[Value]]) -> tuple[Action, ...]:
        """The assignments themselves, in the order `cases` reads them.

        Scoring a set of constraints has to say *which* candidates they got wrong, and a case does not carry the
        action it was read from. Both come from here, so the order is one fact in one place rather than a rule
        two callers have to agree about."""
        names = tuple(sorted(domains))
        return tuple(
            Action(evidence.action, tuple(zip(names, values, strict=True)))
            for values in product(*(tuple(domains[name]) for name in names))
        )

    def of_allowed(self, evidence: Evidence) -> tuple[Example, ...]:
        """The cases for the actions the game lists there, and no others.

        **A whole position for the price of its legal moves.** Reading every candidate is fourteen thousand
        readings of one board and is what learning needs. Asking whether a set of constraints turns away
        something the game allows needs only the moves it allows — twenty or forty of them — and that is a
        different order of cost: the difference between judging a position and learning from one.

        It is what makes looking at many positions affordable, which is what makes choosing between them
        possible at all.

        Every case comes back marked as not holding, because the game listed it: covering one is refusing
        something legal."""
        readings = self.of_state(evidence.where)
        aliases = self._aliases(evidence.where)
        return tuple(
            Example(
                (*readings, *self._of_parameters(one.parameters, aliases, evidence.where)), False, evidence.where
            )
            for one in evidence.legal
        )

    def read(self, state: State, action: Action) -> tuple[Literal, ...]:
        """One candidate in one position, said outright.

        For asking about a single action — running a learned constraint against a move the agent is considering —
        rather than for learning, which reads thousands at once and shares the position between them."""
        return (*self.of_state(state), *self._of_parameters(action.parameters, self._aliases(state), state))

    def of_state(self, state: State) -> tuple[Literal, ...]:
        """The readings that do not depend on the candidate.

        Worked out once for a position and shared by its thousands of candidates. What a position says of itself
        is the same for every move considered in it, and building it again per candidate is the difference between
        reading a position and reading it four thousand times."""
        found: list[Literal] = []
        for name, model in state.models:
            if name in self._paid:
                continue
            if isinstance(model, Grid):
                found.extend(
                    self._said(name, tuple(Number(part) for part in where), value)
                    for where, value in model.items()
                )
            elif isinstance(model, Scalar):
                found.append(self._said(name, (), model.value))
            elif isinstance(model, Machine):
                # Which phase a turn is in, read as a value like any other, so a constraint can be
                # conditioned on it. What a phase offers is the candidates it is asked about, not a reading.
                found.append(self._said(name, (), model.at))
            elif isinstance(model, Map):
                found.extend(self._said(name, (Constant(key),), value) for key, value in model.items)
            elif isinstance(model, List):
                found.extend(
                    self._said(name, (Number(index),), value)
                    for index, value in enumerate(model.items, start=1)
                )
        return tuple(found)

    def _said(self, name: str, naming: tuple[Term, ...], value: Value) -> Literal:
        """What a structure holds at one place, said under that structure's name.

        One reading per place, whatever stands there. A white square holding a white rook reads as
        `grid(1, 1, square(white, piece(white, rook)))` — what is in the cell is one thing, and it is said as one
        term, however deep the thing goes.

        **The places naming the cell are where it is, and the term is what it is.** A square has a colour of its
        own; it has no coordinates, because where it sits is the grid's business. So a reading says where once and
        never again inside the term.

        Saying a thing as one term costs nothing in what can be asked, because a clause puts a variable wherever
        it does not care: `grid(Row, Column, square(Anything, piece(white, Something)))` asks whose the piece on a
        square is and nothing else. Unification goes inside a term, so the parts are as reachable as they would be
        apart, and they arrive already belonging to one thing rather than needing to be kept in step."""
        return Literal(name, (*naming, self._value(value)))

    def _value(self, value: Value) -> Term:
        """What stands somewhere, as a term: a record as a term of its parts, anything else as itself.

        The term is named after what the game called the record's kind, and its arguments are the parts in the
        order the game declared them. A record holding a record goes in as a term holding a term."""
        if isinstance(value, Record):
            return Functor(type(value).__name__.lower(), tuple(self._value(held) for _, held in value.parts))
        return Constant(value)

    def _of_parameters(
        self, parameters: Sequence[tuple[str, Value]], aliases: Mapping[str, Coordinates], state: State
    ) -> tuple[Literal, ...]:
        """Where each of the candidate's parameters points, under the parameter's own name.

        A parameter naming a cell is read as that cell's coordinates rather than as its name, which is what lets
        one parameter's place be tied to another's. Said as the name, `a1` and `a6` are two unrelated things and
        no clause could ever relate them; said as coordinates, what they share is there to be found."""
        found: list[Literal] = []
        for name, value in parameters:
            where = aliases.get(value) if isinstance(value, str) else None
            if where is not None:
                found.append(Literal(name, tuple(Number(part) for part in where)))
            elif isinstance(value, Record):
                found.append(Literal(name, tuple(self._pointed(held) for _, held in value.parts)))
            else:
                found.append(Literal(name, (self._pointed(value),)))
        return (
            *found,
            *self._distances(found),
            *self._pointed_at(found, state),
            *self._from_nothing(found),
            *self._sizes_apart(self._from_nothing(found)),
            *self._lands_on(found, state),
            *self._on_line(found, state),
            *self._belongs(self._pointed_at(found, state), state),
            *self._toward(found, state),
        )

    def _belongs(self, holding: Sequence[Literal], state: State) -> tuple[Literal, ...]:
        """Whose the thing standing where each of the candidate's parameters points is, as a role.

        Read off what `holds` already found rather than pointing again, so the two can never disagree about
        where they are looking.

        **A thing is its owner's where any part of it says so.** The deduction names what it saw standing
        somewhere, which for a game whose things are records is the whole record — a black rook. A game whose
        things are plain values names the value. Both are answered by asking whether the deduction knows this
        thing, and a thing it does not know belongs to nobody, which is a fact about it and not a failure."""
        sides = self.sides
        if not sides or self._acting is None:
            return ()
        acting = self._acting(state)
        found = []
        for one in holding:
            whose = self._owner(sides, one.arguments[-1])
            if whose is None:
                continue
            found.append(
                Literal(BELONGS, (*one.arguments[:-1], Constant(ACTING if whose == acting else ANOTHER)))
            )
        return tuple(found)

    def _owner(self, sides: Sequence[Literal], thing: Term) -> Value | None:
        """The player the deduction says that thing is, or None where it says nothing about it.

        **The thing is the third term and not the last one.** An `owns` fact now ends with how firmly it is
        held, as a `faces` fact has always ended with its step, so reading the last term gives a number where
        a thing was wanted. Named by its place, this reads whichever it is."""
        wanted = thing.value if isinstance(thing, Number) else getattr(thing, "name", None)
        for one in sides:
            if one.predicate != OWNS or len(one.arguments) < 3:
                continue
            said = one.arguments[2]
            if (said.value if isinstance(said, Number) else getattr(said, "name", None)) == wanted:
                return one.arguments[0].name
        return None

    def _toward(self, parameters: Sequence[Literal], state: State) -> tuple[Literal, ...]:
        """How far each of the candidate's numbers goes the way the acting player faces."""
        sides = self.sides
        if not sides or self._acting is None:
            return ()
        acting = self._acting(state)
        facing = next(
            (int(one.arguments[-1].value) for one in sides if one.predicate == FACES and one.arguments[0].name == acting),
            0,
        )
        if not facing:
            return ()
        return tuple(
            Literal(TOWARD, (self._placed(one.predicate, number + 1), Number(int(term.value) * facing)))
            for one in parameters
            for number, term in enumerate(one.arguments)
            if isinstance(term, Number)
        )

    def _on_line(self, parameters: Sequence[Literal], state: State) -> tuple[Literal, ...]:
        """What stands on each cell between where a parameter points and where its steps reach.

        Only where the two lie on a line, which is where every non-nought step is the same size — one of them
        alone is a row or a column, all of them together is a diagonal, and nothing else has cells in between to
        speak of. A step that is not on a line, a knight's, passes over nothing and reads as nothing, which is
        the honest answer rather than an empty one.

        Both pairings of steps to places are offered here as everywhere, since nothing knows which number is
        which axis, and the cases decide which is real."""
        held = [
            (literal.predicate, [one for one in literal.arguments if isinstance(one, Number)])
            for literal in parameters
        ]
        found: list[Literal] = []
        for name, model in state.models:
            if not isinstance(model, Grid):
                continue
            size = len(model.shape)
            for base, places in held:
                if len(places) != size:
                    continue
                others = [
                    (other, number + 1, term)
                    for other, terms in held
                    if other != base
                    for number, term in enumerate(terms)
                ]
                for stepping in permutations(others, size):
                    steps = [int(one[2].value) for one in stepping]
                    far = self._along(steps)
                    if far is None:
                        continue
                    for step in range(1, far):
                        where = tuple(
                            int(places[number].value) + step * (1 if one > 0 else -1 if one < 0 else 0)
                            for number, one in enumerate(steps)
                        )
                        if not model.inside(where):
                            continue
                        found.append(
                            Literal(
                                ON_LINE,
                                (
                                    Constant(base),
                                    *(self._placed(one[0], one[1]) for one in stepping),
                                    Constant(name),
                                    self._value(model.at(where)),
                                ),
                            )
                        )
        return tuple(found)

    def _along(self, steps: Sequence[int]) -> int | None:
        """How far a step goes where it lies on a line, and nothing where it does not.

        On a line means every step that is not nought is the same size. Nothing about a board: two numbers
        changing together at the same rate is what a line is, whatever they are numbers of."""
        sizes = {abs(one) for one in steps if one}
        return sizes.pop() if len(sizes) == 1 else None

    def tied(self, literals: Sequence[Literal]) -> tuple[Literal, ...]:
        """Those readings the candidate is in, out of everything a case carries.

        **What a rule may be built from, as against what it may consult.** A position says sixty-four things
        about a board, and every candidate in that position carries all of them identically — so they separate
        nothing where a rule is grown, while making every clause built from a case a hundred and fifty conditions
        long. What the candidate is in is a couple of dozen readings, and every rule anybody has managed to write
        by hand about how a thing moves uses only those.

        They are not dropped from the case. A rule about a square the candidate never names — something standing
        in the way — needs the board, and a search that may not *start* from the board is not a case that may not
        *mention* it. This is the bottom clause being bounded while the background knowledge keeps everything.

        **What this is called is linkedness**, and it is the standard language bias: a literal is kept where at
        least one of its terms is already bound, a clause is connected where every variable of its head occurs
        in its body. Worked out here rather than declared — Progol and Aleph get the same restriction from input
        modes written by hand, one per predicate, saying that an argument must be bound to a variable already in
        the rule. It is *not* Golem's ij-determinacy, which bounds new variables by there being at most one
        binding for each, and says nothing about sharing a term with the example.

        **It is known to be incomplete, and it is incomplete here.** Bounding a search this way can put the
        answer outside it, and the literature names recursion and predicate invention as what it costs most.
        The rule about something standing in the way is exactly a rule needing a square the candidate does not
        name, and it survives only because `on line` ties those squares back to the candidate's own places. So
        the bound and the vocabulary are one decision: where a rule needs something unconnected, what is wanted
        is a reading that connects it, not a wider search.

        A reading of one place is kept whatever it is: what a scalar says is one thing about the whole position
        and costs nothing to carry."""
        names = {name for name, _, _, _ in self._places}
        return tuple(
            one
            for one in literals
            if one.predicate in TIED or one.predicate in names or len(one.arguments) == 1
        )

    def _sizes_apart(self, sized: Sequence[Literal]) -> tuple[Literal, ...]:
        """How the sizes of the candidate's numbers stand to one another.

        **Without it, no rule that compares two distances is in the space at all.** A constraint is built out of
        the readings a case carried and out of nothing else: `generalised` drops a reading or puts a variable in
        it, and never adds one. So a comparison that is not read is a comparison no clause can ever mention —
        and a thing that travels as far one way as the other is exactly such a comparison. The rule was not hard
        to find, it was absent, which is the same thing that `apart` was added to fix one level down.

        Said in the same shape as any other distance — which two places, which way, how far — so it pairs with
        another the way any two readings pair, and the way already says whether the two are equal, which is the
        whole of the question for anything moving on a diagonal.

        Only sizes of the same kind are compared, as everywhere else: how far a move goes down against how far
        it goes across is two numbers of one kind, where a row against a column is two of different kinds and
        their distance says nothing."""
        held = [
            (one.arguments[0], one.arguments[-1].value)
            for one in sized
            if isinstance(one.arguments[-1], Number)
        ]
        return tuple(
            Literal(
                PLACES_APART,
                (
                    Functor(HOW_FAR, (name,)),
                    Functor(HOW_FAR, (other,)),
                    Constant(self._way(size, elsewhere)),
                    Number(abs(size - elsewhere)),
                ),
            )
            for (name, size), (other, elsewhere) in combinations(held, 2)
            if self._alike_named(name, other)
        )

    def _alike_named(self, one: Term, other: Term) -> bool:
        """Whether two places hold the same kind of thing, asked of the schema rather than of their names.

        It matched strings: "self row" against "self column", recovering from a name what the game had
        declared outright. A place being a term, the parameter and the part are there to be read."""
        kinds = {self._placed(name, number): kind for name, number, _, kind in self._places}
        held, theirs = kinds.get(one), kinds.get(other)
        return held is None or theirs is None or held == theirs

    def _lands_on(self, parameters: Sequence[Literal], state: State) -> tuple[Literal, ...]:
        """What each grid holds where the candidate's numbers reach, taking some of them as a step from another.

        A parameter with as many places as the grid has dimensions can be stepped from; the steps are places of
        the other parameters, in every order, because which step belongs to which dimension is exactly the sort
        of thing there is to find out. A game whose move names two squares gets these too and they mean nothing —
        two squares added together — which costs a few readings and is how nothing has to know which game it is
        reading."""
        held = [
            (literal.predicate, [one for one in literal.arguments if isinstance(one, Number)])
            for literal in parameters
        ]
        found: list[Literal] = []
        for name, model in state.models:
            if not isinstance(model, Grid):
                continue
            size = len(model.shape)
            for base, places in held:
                if len(places) != size:
                    continue
                others = [
                    (other, number + 1, term)
                    for other, terms in held
                    if other != base
                    for number, term in enumerate(terms)
                ]
                for stepping in permutations(others, size):
                    where = tuple(
                        int(places[number].value + one[2].value) for number, one in enumerate(stepping)
                    )
                    reached = model.at(where) if model.inside(where) else OUTSIDE
                    found.append(
                        Literal(
                            LANDS_ON,
                            (
                                Constant(base),
                                *(self._placed(one[0], one[1]) for one in stepping),
                                Constant(name),
                                self._value(reached),
                            ),
                        )
                    )
        return tuple(found)

    def _from_nothing(self, parameters: Sequence[Literal]) -> tuple[Literal, ...]:
        """How far each number the candidate's parameters hold is from nothing, and which way.

        The same shape as a distance between two places — which places, which way, how far — so that a clause
        pairs one of these with another the way it pairs any two readings, and a thing moving by as much one way
        as the other is one variable standing in two distances."""
        return tuple(
            Literal(
                FROM_NOTHING,
                (
                    self._placed(literal.predicate, number + 1),
                    Constant(self._way(0, term.value)),
                    Number(abs(term.value)),
                ),
            )
            for literal in parameters
            for number, term in enumerate(literal.arguments)
            if isinstance(term, Number)
        )

    def _pointed_at(self, parameters: Sequence[Literal], state: State) -> tuple[Literal, ...]:
        """What each structure holds where the candidate's numbers point.

        **One way of reading a grid among several, and not the way.** A move that carries a thing from one square
        to another makes the square it starts from the important one, and in chess or checkers nearly every rule
        is about what stands there. A game that puts a new thing down has no such square: what matters in
        tic-tac-toe or go is the cell being played and whether anything is on it. Both are this reading and
        neither is privileged, because what is offered is every place the candidate's numbers can point at — and
        the whole grid is still read besides, so nothing is hidden by this being here.

        Places are taken across parameters, not only within one. A game whose move is `place(row, column)` points
        at a cell with two parameters of one number each, and a game whose move is `move(origin, destination)`
        points with one parameter of two; reading only the second would be reading only the games that move
        things about."""
        places = [
            (literal.predicate, number + 1, one.value)
            for literal in parameters
            for number, one in enumerate(literal.arguments)
            if isinstance(one, Number)
        ]
        found: list[Literal] = []
        for name, model in state.models:
            if not isinstance(model, Grid):
                continue
            for pointing in permutations(places, len(model.shape)):
                where = tuple(int(one[2]) for one in pointing)
                if not model.inside(where):
                    continue
                found.append(
                    Literal(
                        POINTED_AT,
                        (
                            *(self._placed(held[0], held[1]) for held in pointing),
                            Constant(name),
                            self._value(model.at(where)),
                        ),
                    )
                )
        return tuple(found)

    def _distances(self, parameters: Sequence[Literal]) -> tuple[Literal, ...]:
        """How far apart every two numbers the candidate's parameters hold are.

        **Without these, no rule about how a thing travels is reachable.** A constraint is grown out of the
        readings a case carried and out of nothing else, so a distance that is not read is a distance no clause
        can ever mention — and a bishop, a knight, a king and a pawn are all distances. The engine could compare
        two of them once `apart` existed; this is what puts them in front of the learner.

        **Every pair, not the sensible ones.** A row against a column means nothing in chess and something in
        another game, and OMF has no way to tell which is which. Which of them matter is exactly what there is to
        work out, so all of them are offered and the cases decide.

        The reading names the two places it is about and then says the distance, which is the shape everything
        here relies on: two cases pair because they are about the same two places, and what differs is the
        number."""
        places = [
            (literal.predicate, number + 1, term)
            for literal in parameters
            for number, term in enumerate(literal.arguments)
            if isinstance(term, Number)
        ]
        return tuple(
            Literal(
                PLACES_APART,
                (
                    self._placed(name, place),
                    self._placed(other, elsewhere),
                    Constant(self._way(one.value, two.value)),
                    Number(abs(one.value - two.value)),
                ),
            )
            for (name, place, one), (other, elsewhere, two) in combinations(places, 2)
            if self._alike(name, place, other, elsewhere)
        )

    def _pointed(self, value: Value) -> Term:
        """One of a parameter's places, as a term.

        A number here is a number and not a thing. A parameter is a variable of the constraint problem and what it
        ranges over is what the game declared it ranges over — a row from one to eight, an amount to bet — so
        arithmetic and the comparisons have to reach it. That is the opposite of what stands in a cell, which is a
        thing however it is spelled."""
        if isinstance(value, Record):
            return self._value(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return Constant(value)
        return Number(value)

    def _way(self, one: float, other: float) -> str:
        """Which way the second number lies from the first.

        **Distance alone cannot say which way, and which way is half of every rule about moving.** Two rows one
        apart is a pawn's step and a pawn's step backwards alike, so a game whose pieces go one way has to be said
        twice, once per side — and worse, no variable can join the two halves, because the difference between
        them is a choice of predicate rather than a value. A variable stands for a value.

        Said as a value it can be tied to anything else that is a value, and in particular to whoever is acting.
        Which side goes which way is then a thing the learner can find rather than a thing that must be written
        down twice, and nothing here knows or says what the answer is.

        Named for how the pair stands, not for anything about boards: a game whose numbers are a bet and a pot
        reads the same three answers."""
        return "after" if other > one else "before" if other < one else "alongside"

    def _placed(self, parameter: str, number: int) -> Term:
        """That place of that parameter, as the term it is.

        **A place was a name and is now a term**, which is the whole of this step. `Constant("self row")` was a
        string that spelled out `place(self, row)` and could not be taken apart, so the sort of a place had to
        be recovered by matching the string against the schema, and a drawing saying the same place said it in
        another notation entirely. Said as a term it composes, it unifies, and it is the same object the
        predictor draws with.

        A parameter with one place is that parameter: a number named `x` has no part worth naming, and
        `place(x, x)` would say the same thing twice. The distinction is the one the names already made."""
        held = [one for one in self._places if one[0] == parameter]
        for name, place, part, _ in held:
            if place == number:
                return Constant(parameter) if len(held) == 1 else placed(parameter, part)
        return placed(parameter, str(number))

    def _place(self, parameter: str, number: int) -> str:
        """What that place is called, for anything that needs a name rather than a term."""
        return self._named_place(self._placed(parameter, number))

    def _named_place(self, term: Term) -> str:
        """That place term written as the name it used to be, which is what a reading of a name still expects."""
        if isinstance(term, Constant):
            return str(term.name)
        return f"{part(term, 0)} {part(term, 1)}"

    def _alike(self, parameter: str, number: int, other: str, elsewhere: int) -> bool:
        """Whether two places hold the same kind of thing, which is the only case where their distance means
        anything.

        A row against a column is a number against a number and says nothing about anything — the game declared
        them different kinds, and honouring that is not knowledge smuggled in. Where a game declared no kinds,
        every pair is offered, since nothing is known to be unlike anything else."""
        kinds = {(name, place): kind for name, place, _, kind in self._places}
        held, theirs = kinds.get((parameter, number)), kinds.get((other, elsewhere))
        return held is None or theirs is None or held == theirs

    def _aliases(self, state: State) -> Mapping[str, Coordinates]:
        """What each of the state's cell names points at, gathered once.

        A game that names its cells is asked for those names here rather than each time a candidate mentions one:
        resolving a name walks the game's columns and rows, and a position asks it thousands of times."""
        found: dict[str, Coordinates] = {}
        for _, model in state.models:
            if not isinstance(model, Grid) or model.aliases is None:
                continue
            for where in model.coordinates():
                found[model.aliases.to_alias(where)] = where
        return found

    def _unforeseen(self, evidence: Evidence, domains: Mapping[str, Sequence[Value]]) -> None:
        """Says so where the game lists an action the domains could not have proposed.

        That is not a rule waiting to be learned but a domain that is wrong, and it is worth saying out loud: no
        constraint can ever account for a move that was never a candidate, so a learner left to itself would go on
        failing to explain it for ever without anything saying why."""
        allowed = {name: set(values) for name, values in domains.items()}
        missing = [
            one
            for one in evidence.legal
            if tuple(sorted(name for name, _ in one.parameters)) != tuple(sorted(domains))
            or any(value not in allowed.get(name, ()) for name, value in one.parameters)
        ]
        if missing:
            logger.warning(
                "The game lists %d %s actions no domain could propose, such as %s: the domains are wrong, and no "
                "constraint can account for what was never a candidate",
                len(missing), evidence.action, missing[0].parameters,
            )
