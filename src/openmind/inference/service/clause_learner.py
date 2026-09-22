import logging
import time
from collections.abc import Callable, Sequence

from openmind.inference.model.case_index import CaseIndex
from openmind.inference.model.example import Example
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.model.signature import Signature
from openmind.inference.model.substitution import Substitution
from openmind.inference.service.evaluable_predicates import EvaluablePredicates
from openmind.inference.service.subsumer import Subsumer
from openmind.inference.service.unifier import Unifier
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Term, Variable
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: How a variable the generalising introduces is named.
FRESH = "X{number}"

#: What an exclusion concludes: that the case is refused after all, whatever the clause it hangs off allows.
FORBIDDEN = "refused anyway"


class ClauseLearner:
    """The rules that account for what was seen, learned one at a time.

    **Being legal is not one thing.** One kind of thing may move for reasons another may not, and a thing may have
    two ways of its own — walking somewhere empty, or taking across. A single set of conditions that every case
    satisfies cannot say that, which is why asking what they all have in common gives so little. So a clause is
    grown to cover *some* of the cases, what it covers is set aside, and another is grown for the rest.

    **Each clause is grown from the top.** Start from the clause that says nothing and add the literal that best
    tells the cases where it holds from the cases where it does not, taking the literal's terms from what is
    already bound. Stop when it lets nothing through that it should not.

    **The literals it may add come from the signature and nowhere else** — the game's own readings, with the terms
    they were seen holding. So what OMF can learn to say is exactly what the game let it read, and never something
    somebody thought a game ought to have.

    What it gains over conditions on flattened readings is variables. A reading named
    `"rows from source to target"` carries its slots in its name and can only ever be compared to a value; the
    same reading as `rows(source, target, N)` can tie that number to another reading's, or tie a thing's side to
    the side of whoever is acting. That is one clause where there were two, and it is the reason for all of this.

    It learns four things and is one service: what makes an action legal, what an action leads to and how often,
    how a particular player plays, and what a black box would say. They differ in what the examples are and in
    nothing else.

    It keeps nothing: built once, it is given the examples on every call."""

    def __init__(
        self,
        unifier: Unifier | None = None,
        subsumer: Subsumer | None = None,
        evaluable: EvaluablePredicates | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._unifier = Unifier() if unifier is None else unifier
        self._evaluable = EvaluablePredicates() if evaluable is None else evaluable
        self._subsumer = Subsumer(self._unifier, self._evaluable) if subsumer is None else subsumer
        self._clock = clock

    def learn(
        self,
        examples: Sequence[Example],
        head: Literal,
        budget: InferenceBudget,
        signature: Signature | None = None,
        least: int = 1,
        loosely: float = 0.0,
        positions: int = 1,
        starting: Sequence[Clause] = (),
    ) -> tuple[Clause, ...]:
        """The clauses that account for the cases where it holds, each covering at least `least` of them.

        There is no limit on how many it may take. How many a game needs is a fact about the game and not a number
        to be guessed beforehand, and a limit guessed too low truncates the game: the clauses that would have said
        how something moves are never reached, and what is missing looks like something the learner could not
        learn.

        `loosely` is how much of what holds a clause may wrongly turn away, as a share. None of it by default.
        Allowing some is how the last few stubborn cases get ruled out; allowing too much is how a learned game
        ends up narrower than the real one, which costs moves OMF will now never make.

        `starting` is what is already believed. Given it, a case already accounted for is left alone, and a case
        that is not is first tried as a generalisation of something already held before anything new is begun. That
        is what lets an agent learn as it plays rather than from a corpus: each position it reaches asks only what
        it did not already know, and what it knew is widened rather than replaced. A clause that has stood through
        many positions and still stands is worth more than the same clause found once, and it keeps its identity
        here rather than being rediscovered."""
        deadline = self._clock() + budget.seconds
        holding = [one for one in examples if one.holds]
        against = [one for one in examples if not one.holds]
        index = CaseIndex(tuple(against))
        found: list[Clause] = list(starting)
        left = [one for one in holding if not self.covered(found, one)]
        given = len(found)
        stopped = "every case is accounted for"
        while left:
            if self._clock() >= deadline:
                stopped = "the time ran out"
                break
            grown, used, replaced = self._widened(left, index, head, deadline, loosely, positions, found)
            if replaced is not None:
                found = [one for one in found if one != replaced]
            if grown is None:
                stopped = "nothing further could be generalised"
                break
            covered = [one for one in left if self.covers(grown, one)]
            if len(covered) < least:
                stopped = f"the next clause would account for fewer than {least}"
                break
            found.append(grown)
            left = [one for one in left if one not in covered]
            logger.debug("Kept %s, from %d cases, accounting for %d", grown.readable, len(used), len(covered))
        wrongly = sum(1 for one in against if self.covered(found, one))
        logger.info(
            "Learned %d clauses (%d given) accounting for %d of %d cases, letting %d of %d others through: %s",
            len(found), given, len(holding) - len(left), len(holding), wrongly, len(against), stopped,
        )
        return tuple(found)

    def _widened(
        self,
        left: Sequence[Example],
        index: CaseIndex,
        head: Literal,
        deadline: float,
        loosely: float,
        positions: int,
        held: Sequence[Clause] = (),
    ) -> tuple[Clause | None, list[Example], Clause | None]:
        """One clause, generalised from a case as far as the cases that do not hold allow.

        It starts from a single case said outright, which covers that case and nothing else, and is generalised
        against further cases one at a time. Each generalisation covers strictly more, so the moment one would
        cover a case that does not hold, that generalisation is refused and another case is tried instead.

        This is the opposite of growing a clause downward from nothing, and the difference is where the work goes.
        Growing downward, every candidate literal must be scored against every case that does not hold, and there
        are thousands of candidates. Generalising upward, the cases say what the clause is, and what does not hold
        is consulted once per generalisation rather than once per candidate."""
        allowed = loosely * max(len(index.cases), 1)
        widened = self._widest(left[0], held, index, allowed)
        if widened is not None:
            wider, used, replaced = widened
            return wider, used, replaced
        start = next(
            (number for number, one in enumerate(left) if not self._slips(self._outright(head, one), index, allowed)),
            None,
        )
        if start is None:
            logger.debug("No case can be said outright without letting through what does not hold")
            return None, [], None
        seed = left[start]
        clause = self._outright(head, seed)
        used = [seed]
        for example in (*left[:start], *left[start + 1 :]):
            if self._clock() >= deadline:
                break
            wider = self.generalised(clause, example)
            if wider is None or not wider.body:
                continue
            if self._slips(wider, index, allowed):
                continue
            clause, used = wider, [*used, example]
        if not clause.body or not self._supported(used, positions):
            return None, used, None
        return clause, used, None

    def _widest(
        self, wanted: Example, held: Sequence[Clause], index: CaseIndex, allowed: float
    ) -> tuple[Clause, list[Example], Clause] | None:
        """A clause already believed, widened to account for that case too, where one can be.

        Tried before anything new is begun. A game met a position at a time offers the same rule over and over in
        slightly different circumstances, and a learner that started fresh each time would answer with a new clause
        for each, ending with as many rules as it had seen moves. Widening what is already held keeps one rule where
        there is one rule."""
        for clause in held:
            wider = self.generalised(clause, wanted)
            if wider is None or not wider.body or wider == clause:
                continue
            if self._slips(wider, index, allowed):
                continue
            logger.debug("Widened %s to take in another case", clause.readable)
            return wider, [wanted], clause
        return None

    def _slips(self, clause: Clause, index: CaseIndex, allowed: float) -> bool:
        """Whether that clause lets through more than it may of what does not hold.

        It stops at the first one too many rather than counting them all. Asking how many a clause wrongly covers
        means walking every case that does not hold; asking whether it wrongly covers more than it may usually
        means walking a handful, because a clause generalised too far is caught by nearly the first case it meets.
        On a real game the difference is the difference between answering and not.

        The handful is where the early stop pays, and it is the other case that costs: a clause that lets nothing
        through has to meet every case to prove it, and a clause that lets nothing through is the kind a learner
        keeps. So the cases are narrowed to those sharing a settled reading with the clause before any is asked.
        No case the clause covers is left out of that, since covering means having every settled reading of it."""
        slipped = 0
        settled = [one for one in clause.body if one.ground and not self._evaluable.evaluable(one.predicate)]
        for one in index.narrowed(settled):
            if not self.covers(clause, one):
                continue
            slipped += 1
            if slipped > allowed:
                return True
        return False

    def _outright(self, head: Literal, example: Example) -> Clause:
        """That case said as a clause: everything read of it, and nothing else.

        The most specific clause there is about it. It covers that case and, unless another case reads exactly the
        same, nothing else — which is where generalising starts from."""
        return Clause((head, *(one.denied for one in example.literals if not one.negated)))

    def generalised(self, clause: Clause, example: Example) -> Clause | None:
        """The least general clause covering both what that clause covers and that case.

        This is the whole method in one operation, and there is no search in it. Pair each literal of the clause
        with a reading of the case saying the same sort of thing; where the two agree on a term keep it, and where
        they differ put a variable. What is left covers both, and nothing more general does.

        **The variables come out by construction.** One map from each differing pair of terms to a variable serves
        the whole clause, so a case whose acting player is one side and a case whose acting player is the other
        give the *same* variable wherever either side appeared — in who is acting and in whose the thing is alike.
        That is the clause about a player's own side, arrived at by generalising two cases rather than by searching
        for it, and it is only possible because a reading has arguments to generalise.

        A literal the case says nothing of the same sort about is dropped: it cannot be true of both."""
        if clause.head is None:
            return None
        shared: dict[tuple[Term, Term], Variable] = {}
        by_predicate: dict[str, list[Literal]] = {}
        for literal in example.literals:
            if not literal.negated:
                by_predicate.setdefault(literal.predicate, []).append(literal)
        kept: list[Literal] = []
        for literal in clause.body:
            best: Literal | None = None
            fewest = -1
            for candidate in by_predicate.get(literal.predicate, ()):
                if candidate.arity != literal.arity or not self._about_the_same(literal, candidate):
                    continue
                paired = self._paired_literal(literal, candidate, shared)
                agreeing = sum(1 for one in paired.arguments if not isinstance(one, Variable))
                if agreeing > fewest:
                    best, fewest = paired, agreeing
            if best is not None:
                kept.append(best)
        if not kept:
            return None
        kept = self._speaking(kept)
        if not kept:
            return None
        return Clause((clause.head, *(one.denied for one in kept)), clause.probability, clause.name)

    def _speaking(self, literals: Sequence[Literal]) -> list[Literal]:
        """Those readings that ask for anything at all.

        A reading every one of whose places holds a variable used nowhere else in the clause asks only that the case
        have such a reading — which every case does, since the readings come from one vocabulary. It restricts
        nothing, and a clause carrying a dozen of them is a dozen times slower to ask about and no more selective.

        This is not the same as dropping a reading because the cases suggest it is surplus. It is dropped because it
        cannot say anything, whatever the cases turn out to be, so nothing is risked by its going."""
        counted: dict[Variable, int] = {}
        for literal in literals:
            for variable in literal.variables:
                counted[variable] = counted.get(variable, 0) + 1
        return [
            literal
            for literal in literals
            if not literal.arguments
            or not all(isinstance(one, Variable) and counted.get(one, 0) < 2 for one in literal.arguments)
        ]

    def _about_the_same(self, one: Literal, other: Literal) -> bool:
        """Whether the two readings are about the same thing, so that generalising them means anything.

        A reading names what it is about and then what was read of it: the row *of the target*, the colour *of what
        stands at the source*. The naming places are not values and must not be opened up — generalising the row of
        the target with the row of the source gives "the row of something is something", which is true of every
        case and says nothing about any of them. So two readings pair only where they name the same thing, and only
        what was read of it is generalised."""
        return one.arguments[:-1] == other.arguments[:-1]

    def _paired_literal(
        self, one: Literal, other: Literal, shared: dict[tuple[Term, Term], Variable]
    ) -> Literal:
        """The two readings as one: their terms where they agree, a variable where they do not.

        The same pair of differing terms always gives the same variable, across the whole clause. That is what
        makes one variable appear in two readings rather than two variables appearing in one each, and it is the
        difference between a clause that says something and a clause that says nothing."""
        arguments: list[Term] = []
        for mine, theirs in zip(one.arguments, other.arguments):
            if mine == theirs:
                arguments.append(mine)
                continue
            held = shared.get((mine, theirs))
            if held is None:
                held = Variable(FRESH.format(number=len(shared) + 1))
                shared[(mine, theirs)] = held
            arguments.append(held)
        return Literal(one.predicate, tuple(arguments), one.negated)

    def merged(
        self, clauses: Sequence[Clause], examples: Sequence[Example], loosely: float = 0.0
    ) -> tuple[Clause, ...]:
        """The clauses with any two that say one thing said as one thing.

        Generalising against cases is not enough on its own, and the reason is worth stating. A clause is widened
        to take in a case only when no clause covers that case yet — so once one clause accounts for what one side
        does and another accounts for what the other side does, neither is ever widened again. The set is correct
        and it has learned the same rule twice, once per side, and it will go on learning it once more for every
        further side the game has.

        So the clauses are generalised against *each other*. Two of them anti-unified give the least general clause
        saying what both said; where that lets through nothing it should refuse, it replaces the pair. The two
        side-split clauses become the one clause about a player's own things, which is what there always was."""
        index = CaseIndex(tuple(one for one in examples if not one.holds))
        allowed = loosely * max(len(index.cases), 1)
        found = list(clauses)
        merging = True
        while merging:
            merging = False
            for first in range(len(found)):
                for second in range(first + 1, len(found)):
                    wider = self._both(found[first], found[second])
                    if wider is None or not wider.body:
                        continue
                    if self._slips(wider, index, allowed):
                        continue
                    logger.debug("Merged two clauses into %s", wider.readable)
                    found = [
                        one for number, one in enumerate(found) if number not in (first, second)
                    ] + [wider]
                    merging = True
                    break
                if merging:
                    break
        if len(found) < len(clauses):
            logger.info("Merged %d clauses into %d", len(clauses), len(found))
        return tuple(found)

    def _both(self, one: Clause, other: Clause) -> Clause | None:
        """The least general clause saying what both of those say, or None where they share nothing."""
        if one.head is None or other.head is None or one.head != other.head:
            return None
        shared: dict[tuple[Term, Term], Variable] = {}
        by_predicate: dict[str, list[Literal]] = {}
        for literal in other.body:
            by_predicate.setdefault(literal.predicate, []).append(literal)
        kept: list[Literal] = []
        for literal in one.body:
            best, fewest = None, -1
            for candidate in by_predicate.get(literal.predicate, ()):
                if candidate.arity != literal.arity or not self._about_the_same(literal, candidate):
                    continue
                paired = self._paired_literal(literal, candidate, shared)
                agreeing = sum(1 for held in paired.arguments if not isinstance(held, Variable))
                if agreeing > fewest:
                    best, fewest = paired, agreeing
            if best is not None:
                kept.append(best)
        kept = self._speaking(kept)
        if not kept:
            return None
        return Clause((one.head, *(held.denied for held in kept)), one.probability, one.name)

    def relaxed(
        self, clauses: Sequence[Clause], examples: Sequence[Example], loosely: float = 0.0
    ) -> tuple[Clause, ...]:
        """The clauses with every literal dropped that turns away cases which hold.

        A literal earns its place by ruling something out. One that also rules *in* nothing — that turns away
        cases which hold and meet every other literal of its clause — is not a reason. It is the shape of the
        evidence the clause happened to be grown from: the half of the directions the things in those cases
        happened to go. Growing a clause puts such literals in and nothing ever takes them out, because on the
        cases it was grown from they were never contradicted.

        Dropping goes on while it can, since a literal may only become droppable once another has gone."""
        holding = [one for one in examples if one.holds]
        against = [one for one in examples if not one.holds]
        found: list[Clause] = []
        for clause in clauses:
            dropping = True
            while dropping and clause.body:
                dropping = False
                for literal in clause.body:
                    without = Clause(tuple(one for one in clause.literals if one != literal.denied))
                    if not without.body:
                        continue
                    wrongly = sum(1 for one in against if self.covers(without, one))
                    if wrongly > loosely * max(len(against), 1):
                        continue
                    if sum(1 for one in holding if self.covers(without, one)) <= sum(
                        1 for one in holding if self.covers(clause, one)
                    ):
                        continue
                    clause, dropping = without, True
                    logger.debug("Dropped %s from a clause: it turned away cases that hold", literal.predicate)
                    break
            found.append(clause)
        return tuple(found)

    def excluding(
        self, clauses: Sequence[Clause], examples: Sequence[Example], budget: InferenceBudget
    ) -> tuple[tuple[Clause, tuple[Clause, ...]], ...]:
        """Each clause with what rules out the cases it wrongly covers.

        A clause saying how something may move covers moves the game refuses anyway, and what refuses them is a
        rule in its own right — the way is blocked, what would follow is not allowed. It is learned from the cases
        *that clause* covers and from no others, so what forbids one thing's move can never reach another's. That
        is not tidiness: learned against everything at once, an exclusion picks up conditions that are only ever
        true of the other clauses' cases, and then refuses moves it was never about.

        The exclusions come back beside their clause rather than inside it. A clause with one positive literal is
        a rule with one conclusion, and a denied condition put among its literals would give it two conclusions
        and stop it being one — so the pairing is kept, and what covers is the clause where no exclusion of its
        own covers too."""
        found: list[tuple[Clause, tuple[Clause, ...]]] = []
        for clause in clauses:
            wrongly = [one for one in examples if not one.holds and self.covers(clause, one)]
            if not wrongly:
                found.append((clause, ()))
                continue
            rightly = [one for one in examples if one.holds and self.covers(clause, one)]
            against = [Example(one.literals, True, one.where) for one in wrongly]
            against += [Example(one.literals, False, one.where) for one in rightly]
            learned = self.learn(against, Literal(FORBIDDEN, ()), budget)
            logger.debug("Learned %d ways of being refused for %s", len(learned), clause.readable)
            found.append((clause, learned))
        return tuple(found)

    def allows(self, clauses: Sequence[tuple[Clause, tuple[Clause, ...]]], example: Example) -> bool:
        """Whether any clause covers that case without one of its own exclusions covering it too."""
        return any(
            self.covers(clause, example) and not self.covered(exclusions, example) for clause, exclusions in clauses
        )

    def covers(self, clause: Clause, example: Example) -> bool:
        """Whether that clause's body holds of that case, under some reading of its variables.

        The readings with nothing left open are asked first, and each is one lookup. Nearly every reading of a
        clause generalised from cases is of that sort, and a case that fails any of them fails outright — so almost
        every case is turned away by a handful of lookups and never searched at all. Only what carries variables
        needs trying every way round, and there is little of it."""
        open_ended = []
        for literal in clause.body:
            if literal.ground:
                if self._evaluable.evaluable(literal.predicate):
                    if not self._evaluable.holds(literal):
                        return False
                elif literal not in example.held:
                    return False
            else:
                open_ended.append(literal)
        if not open_ended:
            return True
        return self._holds(open_ended, example, Substitution()) is not None

    def covered(self, clauses: Sequence[Clause], example: Example) -> bool:
        """Whether any of them covers it. A set of clauses accounts for a case where any one of them does."""
        return any(self.covers(one, example) for one in clauses)

    def uncovered(self, clauses: Sequence[Clause], examples: Sequence[Example], by: str) -> dict[Value, int]:
        """The cases that hold and no clause covers, counted by what that reading says of them.

        What a learner has not managed to account for says where to look next. Where what it cannot cover is
        mostly of one kind, what is missing is evidence of that kind — and the way to get it is to go and build
        cases that have it. A learner that only ever sees what a game happens to offer learns what the game
        happens to offer."""
        missing: dict[Value, int] = {}
        for example in examples:
            if not example.holds or self.covered(clauses, example):
                continue
            for literal in example.says(by):
                for argument in literal.arguments:
                    if isinstance(argument, Constant):
                        missing[argument.name] = missing.get(argument.name, 0) + 1
        return dict(sorted(missing.items(), key=lambda held: -held[1]))

    def signature(self, examples: Sequence[Example]) -> Signature:
        """What can be said, read off the cases themselves: every predicate, its arity, and the terms seen at each
        place. Nothing is listed that the cases did not offer, and nothing they offered is left out."""
        arities: dict[str, int] = {}
        values: dict[tuple[str, int], dict[Value, None]] = {}
        numeric: dict[tuple[str, int], bool] = {}
        for example in examples:
            for literal in example.literals:
                arities.setdefault(literal.predicate, literal.arity)
                for place, argument in enumerate(literal.arguments):
                    if not isinstance(argument, Constant):
                        continue
                    values.setdefault((literal.predicate, place), {}).setdefault(argument.name)
                    held = isinstance(argument.name, (int, float)) and not isinstance(argument.name, bool)
                    numeric[(literal.predicate, place)] = held and numeric.get((literal.predicate, place), True)
        return Signature(
            tuple(arities.items()),
            self._evaluable.registered(),
            tuple((name, place, tuple(seen)) for (name, place), seen in values.items()),
            tuple(held for held, was in numeric.items() if was),
        )

    def _supported(self, holds: Sequence[Example], positions: int) -> bool:
        """Whether those cases come from enough different positions to be evidence rather than coincidence.

        Distinct positions, not cases. A clause tying four readings to one variable will hold of plenty of cases
        in the position it was grown from — the readings happened to agree there — and of none anywhere else.
        Counting cases cannot tell that apart from a rule; counting the positions they came from can, because a
        coincidence has to recur somewhere else to survive.

        Where the cases say nothing about where they came from there is nothing to check, and the count of cases
        is all there is to go on."""
        if not holds:
            return False
        places = {one.where for one in holds if one.where is not None}
        return len(places) >= positions if places else len(holds) >= max(1, positions)

    def _holds(self, body: Sequence[Literal], example: Example, agreed: Substitution) -> Substitution | None:
        """A reading of the variables under which every one of those holds of the case, or None."""
        if not body:
            return agreed
        first, rest = agreed.applied(body[0]), body[1:]
        if self._evaluable.evaluable(first.predicate):
            held = self._evaluable.holds(first)
            return self._holds(rest, example, agreed) if held else None
        if first.ground:
            return self._holds(rest, example, agreed) if first in example.held else None
        for literal in example.by_predicate.get(first.predicate, ()):
            found = self._unifier.matches(first, literal)
            if found is None:
                continue
            settled = self._holds(rest, example, agreed.then(found))
            if settled is not None:
                return settled
        return None
