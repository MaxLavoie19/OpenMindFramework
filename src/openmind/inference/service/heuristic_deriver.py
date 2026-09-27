import logging
from collections.abc import Callable, Sequence

from openmind.inference.model.derivation import Derivation
from openmind.inference.model.example import Example
from openmind.inference.model.derivation_step import GIVEN, RESOLVED, DerivationStep
from openmind.inference.model.heuristic import OPTIMAL, Heuristic
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.model.expression import Expression
from openmind.inference.model.parameter import Parameter
from openmind.inference.model.vocabulary import Vocabulary
from openmind.inference.model.worth import Worth
from openmind.knowledge.constant.rule_kind_constant import POSITION
from openmind.inference.service.clause_learner import ClauseLearner
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.side_deducer import OWNS
from openmind.inference.service.subsumer import Subsumer
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Variable
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: What a derived value is called, and what it is about.
AFFORDS = "what {about} affords"
WORTH = "worth of {about}"

#: How many combinations of readings are tried before the count is taken as read.
ENOUGH = 200_000

#: That being able to do less is being worse off — the premise every worth rests on, named so that a worth
#: whose positions did not bear it out can say what it is leaning on rather than leaning on it quietly.
UNSETTLED = "being able to do less is being worse off"


class HeuristicDeriver:
    """What the rules imply about value, said without being asked.

    It is not a question-answering service. Nothing is handed it a list of the things in a game with an
    instruction to price them — it would have to be told what the things are, and being told is the one thing the
    bootstrap cannot rely on. It reads its own rules, finds what they are about, and comes back with what those
    things are worth. The caller learns there are knights from the answer, not before it.

    **How a value is arrived at.** A rule says what a thing may do. Counting the ways it may do it is counting the
    combinations of readings the rule admits — a rule asking for two steps, neither straight nor diagonal, admits
    few; a rule asking only that the way be clear admits many. Nothing is played and no board is needed: the
    readings and the values they were seen taking are enough, so a game with no board yields to the same counting
    as a game with one.

    **And the ordering comes free.** Where everything one thing's rules admit another's admit too, the second
    affords at least what the first does, in every position, always. That is subsumption, and it needs no counting
    at all.

    What comes back is starting points, not answers. A thing whose rules admit fourteen ways is worth looking for
    near fourteen; what it is really worth is what the games say.

    It keeps nothing: built once, it is given the rules on every call."""

    def __init__(
        self,
        subsumer: Subsumer | None = None,
        clause_learner: ClauseLearner | None = None,
        expression_generator: ExpressionGenerator | None = None,
    ) -> None:
        self._subsumer = Subsumer() if subsumer is None else subsumer
        self._learner = ClauseLearner() if clause_learner is None else clause_learner
        self._generator = ExpressionGenerator() if expression_generator is None else expression_generator

    def derive(
        self,
        clauses: Sequence[Clause],
        cases: Sequence[Example],
        budget: InferenceBudget,
    ) -> tuple[Heuristic, ...]:
        """Everything the rules say about what things are worth.

        Nothing is asked for and nothing is named. What comes back is one heuristic per thing the rules turn out
        to be about, each with the value its own rules admit and the reasoning that reached it."""
        about = self._about(clauses)
        found: list[Heuristic] = []
        for thing, mine in sorted(about.items(), key=lambda held: str(held[0])):
            admitted = sum(1 for one in cases if any(self._permits(clause, one) for clause in mine))
            found.append(
                Heuristic(
                    thing,
                    mine[0],
                    POSITION,
                    self._why(thing, mine, admitted),
                    aim=OPTIMAL,
                    parameters=(Parameter(WORTH.format(about=thing), float(admitted), AFFORDS.format(about=thing)),),
                )
            )
        logger.info(
            "The rules are about %d things, and say what each affords: %s",
            len(found),
            "; ".join(one.readable for one in found),
        )
        return tuple(found)

    def afforded(
        self,
        clauses: Sequence[Clause],
        cases: Sequence[Example],
        covers: Callable[[Clause, Example], bool],
        about: Callable[[Sequence[Literal]], Sequence[Literal]] | None = None,
    ) -> tuple[Heuristic, ...]:
        """What each thing affords, where the rules say what is *refused* rather than what is allowed.

        **The same question, and the arithmetic runs the other way.** `derive` counts the cases a thing's rules
        permit, because a rule there says what may be done. A game learned as refusals has no such rule: what a
        thing may do is whatever nothing refuses, so the count is of cases no constraint covers. Handed
        refusals, `derive` would count the cases some constraint *matches* — and a thing hemmed in by more rules
        would come out worth more, which is the answer exactly inverted.

        **And what a case is about is read from the case, not from the rules.** A refusal is usually about no
        particular thing — "you may not move off the board" names none — so grouping by what the rules pin down
        finds almost nothing. What a case is about is what its own readings say stands where it acts, which is
        there in every case whether any rule mentions it or not.

        **`about` says which of a case's readings the case is *about*, and without it this measures nothing.**
        A case carries the whole board — every square of every structure — so every piece standing anywhere
        counts in every case, and what comes back is the number of cases that survived, said once per thing.
        Measured against chess it made all seventeen things worth 1724 apiece, which is the same defect as
        reading a whole position to find what a move moves. What a case is about is what its own parameters
        point at, which the caller's vocabulary already works out for bounding its search.

        What comes back is starting points, exactly as `derive`'s are: a thing that can do fourteen things here
        is worth looking for near fourteen, and what it is really worth is what the games say."""
        standing: dict[Value, int] = {}
        seen: dict[Value, int] = {}
        for case in cases:
            refused = any(covers(clause, case) for clause in clauses)
            for thing in self._carried(about(case.literals) if about else case.literals):
                seen[thing] = seen.get(thing, 0) + 1
                if not refused:
                    standing[thing] = standing.get(thing, 0) + 1
        found = [
            Heuristic(
                thing,
                clauses[0] if clauses else Clause(()),
                POSITION,
                self._why(thing, (), standing.get(thing, 0)),
                aim=OPTIMAL,
                parameters=(
                    Parameter(WORTH.format(about=thing), float(standing.get(thing, 0)), AFFORDS.format(about=thing)),
                ),
            )
            for thing in sorted(seen, key=str)
        ]
        logger.info(
            "Of %d candidates, %d survive the rules; %d things can do anything at all: %s",
            len(cases),
            sum(1 for case in cases if not any(covers(clause, case) for clause in clauses)),
            sum(1 for one in found if one.parameters[0].initial),
            "; ".join(one.readable for one in found if one.parameters[0].initial),
        )
        return tuple(found)

    def _carried(self, literals: Sequence[Literal]) -> tuple[Value, ...]:
        """What a case is about: the things its own readings say are there.

        Read off the case rather than declared, so a game whose things are records says records and one whose
        things are letters says letters. A number is not one of them — a coordinate is where something is and
        never what it is."""
        found: dict[Value, None] = {}
        for literal in literals:
            if not literal.arguments:
                continue
            last = literal.arguments[-1]
            if isinstance(last, Functor) or (isinstance(last, Constant) and isinstance(last.name, str) and last.name):
                found[last if isinstance(last, Functor) else last.name] = None
        return tuple(found)

    def holdings(self, worth: Worth) -> tuple[Heuristic, ...]:
        """What a reasoner's worths say, as heuristics that know which structure each thing was found in.

        **The base is the thing that makes a seed buildable.** Everything else here gives back a bare value — a
        knight, with nothing saying which of a position's structures a knight stands in — and turning that into
        an expression means guessing. A worth already carries the model it was read from, so this hands it on.

        **A worth that rests on nothing says so.** `WorthReasoner` checks whether the positions it saw bore out
        that being able to do less is being worse off, and where they did not it counts the worths anyway and
        reports them as resting on nothing. That is a premise holding only some of the time, which is what
        `chances` is for — so the reasoning can be argued with rather than the number being quietly trusted."""
        settled = () if worth.settled else (self._unsettled(worth),)
        return tuple(
            Heuristic(
                value,
                self._holding(model, value),
                POSITION,
                Derivation(self._holding(model, value), (), settled),
                aim=OPTIMAL,
                parameters=(Parameter(WORTH.format(about=value), held, model),),
            )
            for model, value, held in worth.holdings
        )

    def seeds(
        self,
        heuristics: Sequence[Heuristic],
        vocabulary: Vocabulary,
        sides: Sequence[Literal] = (),
    ) -> tuple[tuple[Expression, float], ...]:
        """Those heuristics as expressions the search can start from, each with the weight the rules imply.

        **The weight is not a number carried alongside the shape — it is the shape's weight.** A linear
        heuristic values a position as the sum of its rules' weighted readings, so the weight on "how many
        knights I have" *is* what a knight is worth. The engine's count goes in as that term's starting weight
        and the fitter moves it.

        **The vocabulary is needed and the specification did not say so.** Resolving a thing to the structure it
        stands in, writing a player's name as `me` rather than as itself, and knowing how many coordinates a
        place has are all things only the vocabulary holds. A heuristic whose parameter names its base is
        resolved outright; one that does not is offered against every base whose values include it, and the
        price sorts them out — declining to offer them would be judging a candidate by how sensible it looks,
        which is the one filter this is not allowed.

        **What a seed buys, said plainly.** Not immunity: a term whose gradient does not pay is shrunk to
        nothing in one step whatever weight it started at, and that is the filter working. What it buys is being
        *expanded first* — its conditions, its thresholds and its children are generated before anything else's,
        so the term that is really wanted is reached in the generation the search would otherwise spend
        rediscovering it."""
        owned = {
            str(one.arguments[1].name): str(one.arguments[0].name)
            for one in sides
            if one.predicate == OWNS and len(one.arguments) > 2
        }
        found: dict[str, tuple[Expression, float]] = {}
        for heuristic in heuristics:
            if not heuristic.parameters or not heuristic.parameters[0].initial:
                continue
            weight = heuristic.parameters[0].initial
            for base in self._bases(heuristic, vocabulary):
                owner = owned.get(base) or self._generator.owning(base, vocabulary)
                for expression in self._generator.counting(base, heuristic.about, vocabulary, owner):
                    found.setdefault(expression.template, (expression, weight))
        logger.info(
            "The rules seed the search with %d terms from %d things worth something: %s",
            len(found),
            sum(1 for one in heuristics if one.parameters and one.parameters[0].initial),
            "; ".join(f"{one.template} at {weight:g}" for one, weight in list(found.values())[:6]),
        )
        return tuple(found.values())

    def _bases(self, heuristic: Heuristic, vocabulary: Vocabulary) -> tuple[str, ...]:
        """Which structures that heuristic's thing might stand in: the one it names, or every one holding it."""
        named = heuristic.parameters[0].holds if heuristic.parameters else ""
        if named and named in vocabulary.indices_by_base:
            return (named,)
        return tuple(
            base for base, values in sorted(vocabulary.values_by_base.items()) if heuristic.about in values
        )

    def _holding(self, model: str, value: Value) -> Clause:
        """That a thing of this kind stands in this structure, as a clause, so the worth has logic behind it."""
        return Clause((Literal(model, (Constant(value),)),))

    def _unsettled(self, worth: Worth) -> Clause:
        """That being able to do less is being worse off, which the positions seen did not bear out."""
        return Clause((Literal(UNSETTLED, (Constant(worth.ended),)),))

    def ordered(self, clauses: Sequence[Clause]) -> tuple[tuple[Value, Value], ...]:
        """Which things afford at least what another does, from the rules alone and no counting.

        Where everything one thing's rules allow another's allow too, the second affords at least what the first
        does — in every position, always, and before anything is looked at."""
        about = self._about(clauses)
        found = self._subsumer.ordered(
            {thing: tuple(self._besides(one, thing) for one in mine) for thing, mine in about.items()}
        )
        for wider, narrower in found:
            logger.info("Whatever %s may do, %s may do: it affords at least as much", narrower, wider)
        return found

    def _besides(self, clause: Clause, thing: Value) -> Clause:
        """That rule with the reading naming the thing set aside.

        What a thing affords and when it may do it are different questions, and the reading saying which thing is
        standing there answers the second. Left in, it makes every thing incomparable with every other — each
        rule asks for its own thing and no other rule asks for that — so the ordering that ought to fall out of
        the rules falls out of nothing. Set aside, what is left is what the thing may do, which is the question."""
        return Clause(
            tuple(
                one
                for one in clause.literals
                if not (one.arguments and one.arguments[-1] == Constant(thing))
            ),
            clause.probability,
            clause.name,
        )

    def _about(self, clauses: Sequence[Clause]) -> dict[Value, list[Clause]]:
        """What the rules turn out to be about, and which rules are about each.

        A rule is about whatever it pins down and does not vary: the kinds named in it that the reading's value
        place holds outright. Nothing is declared and nothing is looked up — a game whose rules pin down no kind
        simply has one thing, which is the game."""
        found: dict[Value, list[Clause]] = {}
        for clause in clauses:
            for thing in self._named(clause):
                found.setdefault(thing, []).append(clause)
        return found

    def _named(self, clause: Clause) -> tuple[Value, ...]:
        """The kinds a rule pins down: the values it asks for outright, where other rules vary them.

        The value place of a reading is its last, since a reading names what it is about and then what was read of
        it. A rule asking for that place to hold one particular thing is a rule about that thing."""
        found: dict[Value, None] = {}
        for literal in clause.body:
            if not literal.arguments:
                continue
            last = literal.arguments[-1]
            if isinstance(last, Constant) and isinstance(last.name, str) and last.name:
                found.setdefault(last.name)
        return tuple(found)

    def _permits(self, clause: Clause, case: Example) -> bool:
        """Whether the rule allows that way of acting, its readings taken together."""
        return self._learner.covers(clause, case)

    def _why(self, thing: Value, clauses: Sequence[Clause], admitted: int) -> Derivation:
        """The reasoning that reached it: the rules it read, and the counting it did over them."""
        steps = tuple(
            DerivationStep(number + 1, GIVEN, (), clause) for number, clause in enumerate(clauses)
        )
        reached = Clause(
            (Literal(WORTH.format(about=thing), (Constant(thing), Constant(admitted))),),
            1.0,
            AFFORDS.format(about=thing),
        )
        return Derivation(
            reached,
            (*steps, DerivationStep(len(steps) + 1, RESOLVED, tuple(range(1, len(steps) + 1)), reached)),
        )
