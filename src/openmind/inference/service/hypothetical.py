import logging
from collections.abc import Callable, Mapping, Sequence

from openmind.inference.model.evidence import Evidence
from openmind.inference.model.example import Example
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.predictor.service.consequence_drawer import ConsequenceDrawer
from openmind.world.service.changer import Changer
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.moment import HAPPENS, after
from openmind.statement.model.term import Constant, Functor, Number, Term
from openmind.structure.model.grid import Grid
from openmind.structure.model.record import Record
from openmind.structure.model.schema import ActionKind
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.statement.model.change import LOSING
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: A condition asking whether some other action would be allowed by the rules below the one asking.
#:
#: Its arguments are that action's parameters, in the order the readings say them: each parameter's places in
#: turn, the parameters taken by name in order. So a game whose move is an origin and a destination asks about
#: four numbers, and a game whose action is a bet asks about one.
ALLOWED = "allowed by the rules below"

#: The same question, asked of the position the candidate leads to rather than the one it is played in.
#:
#: **The last rule of chess that could not be written.** A king may not be left where it can be taken, and that
#: is a fact about a board that does not exist yet: every reading is of the board as it stands, so no clause
#: over them can reach it. What is needed is not another reading but the *other half of the design* — the
#: predictor says what the candidate does, the changes are made, and the question is put to the board that
#: results.
#:
#: It is the first thing that needs both halves at once. Until now the predictor learned what a move does and
#: the constraints learned what is refused, and neither used the other.
ALLOWED_AFTER = "allowed by the rules below, once this is done"

#: Whether, once this candidate is done, the other side has an action allowed by the rules below whose doing
#: takes away a thing of this kind belonging to this player.
#:
#: Its arguments are what the change would befall: whose the thing is, and what it is — said in the values the
#: game declared, never as a square.
#:
#: **Said as what could happen, because "lands on a place" is a chess rule wearing a general coat.** Arriving
#: somewhere is the danger in chess and in checkers, and it is not the danger in a game where a card is turned,
#: a score passes a mark, or a piece is flipped. What is general is that something could *happen* next that one
#: would rather did not, and happenings already have a language here: the changes the predictor learns an action
#: brings about.
#:
#: **It composes three things rather than adding a fourth.** The board this candidate leads to, from the
#: predictor. Whether an action is allowed there, from the layering that keeps the question from running away.
#: And what that action would do, which is what the predictor has been learning all along. So the game-specific
#: part shrinks to naming which change is the bad one, in the game's own declared words.
TAKEN_AFTER = "taken from that player, once this is done, by something they could then do"

#: A thing of somebody's being taken, as the kind of happening it is.
#:
#: **What a search can say, where the question above is only something it can be asked.** `TAKEN_AFTER` answers
#: yes or no and nothing can look inside it: it cannot be composed with another condition, denied, generalised,
#: or proposed by a search that has never seen one. Said as a happening at a moment it is an ordinary reading,
#: and a case that carries it is matched by an ordinary lookup — a moment being part of what a literal is.
#:
#: Said as *taken* and not as *absent* because a constraint's body is positive: every condition in it is
#: something that must hold, so "my king is not there afterwards" cannot be said at all. That is what this
#: layer's own note meant — "what is general is that something could happen next that one would rather did
#: not, and happenings already have a language here."
TAKEN = "taken"

#: The candidate this case is about, as the thing a later moment is later than. A case is about one candidate,
#: so "once this has happened" needs no more of a name than that.
THIS = "this"


class Hypothetical:
    """A move nobody is making, asked about as though somebody were.

    **Some rules are about what could happen rather than what is happening.** A king may not castle across a
    square an enemy could reach; a move may not leave the king where it could be taken. Neither is a fact about
    the position as it stands — both are about an action nobody has proposed, asked of the same position.

    **Which rules answer it is not a free choice.** Asking whether the enemy's move is allowed, by all the rules,
    means asking whether it would leave *their* king safe, which asks about my moves, for ever. The question has
    to be put to strictly fewer rules than the rule putting it — and the rules that ask no such question are
    exactly the ones that can be asked. That is where the layers come from, and nothing declares them: a
    constraint that asks about a hypothetical is above every constraint that does not, by the fact of asking.

    It keeps nothing: built once, it is given the position and the action on every call."""

    def __init__(
        self,
        readings: CandidateReadings,
        action: ActionKind,
        drawer: ConsequenceDrawer | None = None,
        changer: Changer | None = None,
        doing: Sequence | Callable[[], Sequence] = (),
        players: Sequence[Value] = (),
        domains: Mapping[str, Sequence[Value]] | None = None,
        acting: Callable[[State], Value] | None = None,
        holds: Callable[[Sequence[Clause], Example], bool] | None = None,
    ) -> None:
        self._readings = readings
        self._action = action
        self._drawer = drawer
        self._changer = changer
        # What the predictor has concluded an action does — or a way of asking for it, which is what a caller
        # still learning wants. The two halves learn at once: what a move does is being worked out while what
        # is refused is being worked out, and a constraint reaching past this board reaches it through
        # drawings that changed since this was built. Handed over as a value, it is the drawings of the
        # position this was constructed in, which for a loop built before its first position is none at all —
        # and a question that can never be answered is answered no, silently, in every position.
        self._doing = doing
        self._players = players
        self._domains = domains
        # Whose turn it is, read from the position by whoever knows which of its structures says so. OMF
        # cannot know that: a game declares a grid and some scalars, and which scalar is the turn is not a
        # fact it hands over. Taking the first player given was wrong the moment a consequence depended on
        # who acted — `Other`, or a value drawn from the actor — and it failed silently, as a drawing that
        # would not narrow.
        self._acting = acting
        # How to ask whether a consequence's conditions hold, handed over rather than held: answering one
        # means putting a clause to a case, which is the refusal learner's work, and the learner holds
        # this — so keeping a learner here would close a ring. Given none, every consequence draws, which
        # is what it did before anything asked.
        self._holds = holds

    @property
    def doing(self) -> Sequence:
        """What the predictor says an action does, as it stands now."""
        return self._doing() if callable(self._doing) else self._doing

    def after(self, example: Example) -> State | None:
        """The position that case's candidate leads to, or None where nothing can say.

        **What the predictor is for, used by the constraints for the first time.** A consequence says where its
        parts come from rather than what they are, so drawing it against this position and this candidate is
        what turns what a move does in general into what it does here. Where the predictor has concluded
        nothing, or cannot draw a consequence on this board, there is no position to ask about — and no answer
        is given rather than a wrong one."""
        doing = self.doing
        if self._drawer is None or self._changer is None or not doing:
            return None
        action = self.acting(example)
        if action is None:
            return None
        acting = self._acting(example.where) if self._acting else None
        # Drawn in the order the game made them, and only where their conditions hold — the case to ask about
        # being this very example, which is what it is. Without asking, every move takes something and clears
        # every castling right, which is a wrong board rather than a missing one.
        changes = self._drawer.changes(
            doing, example.where, action, acting, self._players, example, self._holds
        )
        return self._changer.applied(example.where, changes) if changes else None

    def askable(self, example: Example) -> tuple[Literal, ...]:
        """The questions about what this candidate could cost, that a search may put into a body.

        **What could be taken was answerable and unaskable.** A clause carrying one of these is checked
        correctly and has been since it was written; nothing ever built one. A body is assembled out of the
        readings a case *carries*, and this is not a reading — it is a question you can put to a case, about a
        board that does not exist yet. So the one rule this whole layer was built for could be written by hand
        and never found, and a search given every other rule of chess for free, both the moves it should refuse
        and the moves it must not, came back with the halfmove clock and the colour of a square.

        **One per kind of thing the game has, and whose it is is whoever is acting here.** Nothing says which
        kind matters: losing a pawn is perfectly legal and losing the king is not, and which is which is what
        the guard decides. Said ground, with this case's own mover, because that is how every other rule about
        ownership has been found — ground per case, and turned into a variable by widening across cases whose
        movers differ. It comes out as `turn(X), taken from X once this is done ... king`, which is the shape
        the rule was written by hand in.

        Empty where the question cannot be put at all: without a predictor there is no board to ask about, and
        without knowing who acts there is nobody to ask it for."""
        if self._drawer is None or self._acting is None or example.where is None:
            return ()
        whose = self._acting(example.where)
        if whose is None:
            return ()
        return tuple(
            Literal(TAKEN_AFTER, (Constant(whose), Constant(one))) for one in self._kinds(example.where, whose)
        )

    def _kinds(self, state: State, whose: Value) -> tuple[Value, ...]:
        """What kinds of thing could be taken here, as the game declares them.

        Read off the position rather than declared, since a game says what its values are by having them. The
        players' own names are dropped: a record says whose a thing is and what it is in the same breath, and
        `whose` is already being said by the other argument.

        **Only what something has been seen to be lost from.** Chess keeps its squares' colours on a grid
        beside its pieces, so asking every grid gives "taken from white, once this is done: a light square" — a
        question nothing can ever answer yes, because nothing takes a square's colour away. It would never
        stand as a constraint, having refused nothing, but it costs a whole pass over the board after every
        candidate to find that out each time. Which models a thing can be lost from is what the predictor has
        already worked out, and it is read here in the vocabulary the changes themselves declare — so a predictor
        saying a capture one way and a predictor saying it another are asked the same questions."""
        emptied = {one.model for one in self.doing if one.change in LOSING}
        found = dict.fromkeys(
            part
            for name, model in state.models
            if isinstance(model, Grid) and name in emptied
            for standing in model.cells
            if standing is not None
            for part in ([held for _, held in standing.parts] if isinstance(standing, Record) else [standing])
        )
        return tuple(one for one in found if one != whose and one not in self._players)

    def taken(self, example: Example, whose: Value, what: Value, refuses: Callable[[Example], bool]) -> bool:
        """Whether the other side, once this candidate is done, has an allowed action that takes away a thing of
        theirs to lose — said as a change and never as a square.

        One ply and no more. The board this leads to is worked out, every candidate is read in it, those the
        rules below refuse are set aside, and what is left is asked what it would *do*. A removal of the thing
        named is the answer; anything else is not.

        **The rules below and not all of them**, for the reason the layering exists: asking whether their reply
        is allowed by every rule would ask whether it leaves *their* king safe, which asks about mine, for ever.

        False where nothing can say — no predictor, no domains, a board that will not draw. A question that
        cannot be put is not a question answered yes, and here answering yes would refuse a move that is
        perfectly legal."""
        if self._domains is None:
            return False
        after = self.after(example)
        if after is None:
            return False
        theirs = self._acting(after) if self._acting else next(
            (one for one in self._players if one != whose), None
        )
        for candidate in self._readings.candidates(Evidence(after, self._action.name, ()), self._domains):
            # What it would do is asked first, and whether it is allowed second. Nearly every candidate takes
            # nothing of interest, and finding that out is drawing a consequence and looking at a square; asking
            # whether it is allowed means reading a whole case and putting it to every rule below. Asked the
            # other way round this is minutes a position rather than moments.
            if not self._takes(after, candidate, whose, what, theirs):
                continue
            if not refuses(Example(self._readings.read(after, candidate), False, after)):
                return True
        return False

    def happenings(self, example: Example, refuses: Callable[[Example], bool]) -> tuple[Literal, ...]:
        """What could be taken from the mover once this candidate is done, as readings of the moment it happens.

        **The same work the question does, handed over as vocabulary instead of as an answer.** `taken` asks
        about one kind and says yes or no; this walks the board the candidate leads to once and says every kind
        that could go, as `happens(taken(Whose, what), once this has happened)`. A case carrying those is
        matched by lookup like any other reading, so a clause may mention one, deny the rest of its body around
        it, generalise it, or be grown with it — none of which an answer permits.

        One pass and not one per kind. Asked a kind at a time, chess walks fourteen thousand candidates six
        times over to learn what one walk would have told it.

        Empty where the question cannot be put: no predictor, nobody known to be acting, a board that will not
        draw. A question that cannot be put is not a question answered yes."""
        if self._domains is None or self._acting is None or example.where is None:
            return ()
        board = self.after(example)
        if board is None:
            return ()
        whose, theirs = self._acting(example.where), self._acting(board)
        found: set[Value] = set()
        for candidate in self._readings.candidates(Evidence(board, self._action.name, ()), self._domains):
            taking = self._taking(board, candidate, whose, theirs)
            if not taking or taking <= found:
                continue
            if refuses(Example(self._readings.read(board, candidate), False, board)):
                continue
            found |= taking
        when = after(Constant(THIS))
        return tuple(
            Literal(HAPPENS, (Functor(TAKEN, (Constant(whose), Constant(one))),), when=when)
            for one in sorted(found, key=repr)
        )

    def _taking(self, state: State, action: Action, whose: Value, theirs: Value) -> set[Value]:
        """Every kind of thing of that player's that doing this there would take away.

        The set rather than one answer, so a walk of the board says everything it found rather than being asked
        again per kind."""
        if self._drawer is None:
            return set()
        found: set[Value] = set()
        for one in self.doing:
            change = self._drawer.drawn(one, state, action, theirs, self._players)
            at = getattr(change, "losing", None)
            if at is None:
                continue
            held = state.model(change.model)
            standing = held.at(at) if isinstance(held, Grid) and held.inside(at) else None
            if standing is None or standing == getattr(change, "value", None):
                continue
            parts = [one for _, one in standing.parts] if isinstance(standing, Record) else [standing]
            if whose in parts:
                found.update(one for one in parts if one != whose)
        return found

    def _takes(self, state: State, action: Action, whose: Value, what: Value, theirs: Value) -> bool:
        """Whether doing that there takes away a thing of that player's, of that kind.

        **A consequence is drawn with its conditions unasked, and then asked before it is believed.** Drawing is
        cheap and asking is not — answering a condition means reading a whole case — so the order is: draw,
        look at the square, and only where something of the wanted kind stands there go and find out whether
        that consequence applies at all. Over fourteen thousand candidates nearly none reaches the last step.

        **Without the last step a knight takes a pawn in passing.** A consequence's square is drawn from the
        action, so one that belongs to a pawn's capture still computes a square for a knight's move — and the
        square it computes is a real square with a real piece on it. Measured: a knight on h7 going to f6 was
        reported as taking the king on f7, by the consequence for taking in passing and by the one that moves
        a castling's rook, neither of which a knight has anything to do with. King safety then refused a legal
        move, the repair went to mend a rule that was not wrong, and a night's training sat in it."""
        if self._drawer is None:
            return False
        case = None
        for one in self.doing:
            change = self._drawer.drawn(one, state, action, theirs, self._players)
            at = getattr(change, "losing", None)
            if at is None:
                continue
            held = state.model(change.model)
            standing = held.at(at) if isinstance(held, Grid) and held.inside(at) else None
            if standing is None or standing == getattr(change, "value", None):
                continue
            if not self._is(standing, whose, what):
                continue
            if not one.when or self._holds is None:
                return True
            # Read once and only here: a consequence with no conditions happens every time, and one with
            # conditions is worth the reading only now that its square is known to hold what is being asked
            # about.
            case = case or Example(self._readings.read(state, action), False, state)
            if self._holds(one.when, case):
                return True
        return False

    def _is(self, standing: Value, whose: Value, what: Value) -> bool:
        """Whether what stands there is that player's and of that kind, whatever the game calls those parts."""
        parts = [held for _, held in standing.parts] if isinstance(standing, Record) else [standing]
        return whose in parts and what in parts

    def acting(self, example: Example) -> Action | None:
        """The candidate that case was read from, gathered back out of its own readings.

        A case does not carry the action it is about, and it does not need to: each parameter is read under its
        own name, so the action is there to be put back together. Reading it out beats handing it about, which
        would mean every caller of every method in between carrying something only this one needs."""
        held = {one.predicate: one.arguments for one in example.literals if not one.negated}
        gathered: list[tuple[str, Value]] = []
        for name, kind in sorted(self._action.parameters, key=lambda one: one[0]):
            arguments = held.get(name)
            if arguments is None:
                return None
            settled = [self._plain(one) for one in arguments]
            if any(one is None for one in settled):
                return None
            parts = len(getattr(kind, "parts", ()) or ())
            if getattr(kind, "builds", None) is not None and parts:
                gathered.append((name, kind.builds(*settled)))
            else:
                gathered.append((name, settled[0]))
        return Action(self._action.name, tuple(gathered))

    def case(self, state: State, values: Sequence[Term]) -> Example | None:
        """That hypothetical action in that position, read as a case, or None where the terms do not make one.

        The terms are the action's parameters laid end to end, in the order the readings say them, and they are
        gathered back into parameters here. A term standing for nothing settled makes no action, and the question
        goes unanswered rather than being answered wrongly."""
        gathered: list[tuple[str, Value]] = []
        left = list(values)
        for name, kind in sorted(self._action.parameters, key=lambda held: held[0]):
            parts = len(kind.parts)
            taken, left = left[: parts or 1], left[parts or 1 :]
            if len(taken) < (parts or 1):
                return None
            settled = [self._plain(one) for one in taken]
            if any(one is None for one in settled):
                return None
            if kind.builds is not None and parts:
                gathered.append((name, kind.builds(*settled)))
            else:
                gathered.append((name, settled[0]))
        if left:
            return None
        action = Action(self._action.name, tuple(gathered))
        return Example(self._readings.read(state, action), False, state)

    def asked(self, clause: Clause) -> bool:
        """Whether that constraint asks about a hypothetical, which is what puts it above those that do not.

        Every kind of asking counts. A rule about the board a move leads to is as much above the plain rules as
        one about this board, and treating it as though it asked nothing would let it be used to answer itself —
        whether the king is safe after my move would depend on whether it is safe after theirs, for ever.

        `TAKEN_AFTER` was missing from here while it was the only one of the three no rule had yet been written
        in, so nothing showed. It asks two questions rather than one — what the other side could do, and whether
        they are allowed to do it — which makes leaving it out of the layering worse than the others, not
        better."""
        return any(one.predicate in (ALLOWED, ALLOWED_AFTER, TAKEN_AFTER) for one in clause.body)

    def below(self, clauses: Sequence[Clause]) -> tuple[Clause, ...]:
        """Those of them that may answer such a question: the ones that ask none themselves.

        Worked out from the constraints and never declared. A game that says how its pieces travel and a game
        that also says a king may not be left attacked are the same game to OMF; which of its rules is the
        further one is a fact about what those rules refer to."""
        return tuple(one for one in clauses if not self.asked(one))

    def _plain(self, term: Term) -> Value | None:
        if isinstance(term, Number):
            return term.value
        if isinstance(term, Constant) and not isinstance(term.name, Record):
            return term.name
        return None
