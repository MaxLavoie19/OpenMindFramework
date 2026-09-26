import collections
import logging
import math
from collections.abc import Sequence

from openmind.language.model.correspondence import Correspondence
from openmind.language.model.coupling import Coupling
from openmind.inference.service.information import Information
from openmind.inference.service.statistics import Statistics
from openmind.language.model.grammar import Grammar
from openmind.language.model.happening import Happening
from openmind.language.model.role import Role
from openmind.language.model.shape import Shape
from openmind.structure.model.value import Value
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: One shape's sightings: the notation, and what happened said in parts.
Sighted = tuple[str, dict[str, Value]]


class CouplingLearner:
    """What each place of each shape of a game's notation says about what happened.

    **A symbol is a symbol when it tells us something.** What is measured is how much knowing that a character
    stands in a place narrows one part of the happening. A character that narrows nothing is not a symbol however
    often it turns up, so nothing has to decide beforehand what any of it means.

    **And however seldom.** How much a symbol says is asked where it stands, not averaged over the sightings it
    is absent from — those say how much attending to it is worth across a corpus, which is a fact about how
    often the thing is spoken of rather than about what the word means. Averaged, a rare symbol cannot clear any
    threshold whatever it says, and every file letter in chess sat exactly at that ceiling.

    **It measures places of shapes, not counts from the start of a string.** A notation's length moves its
    places about, so `e` opens both `e4` and `exd5` while saying where the move lands in one and where it started
    in the other. A shape fixes the layout, and only then is a place worth measuring. This is why the syntax has
    to be learned first, and why it is learned from the notations alone: a place measured against what happened
    would agree with the happenings by construction.

    **It also finds what the notation leaves out.** A part that no symbol tells us anything about is a part the
    notation does not carry — and in chess that is where a move *starts*, which is left out precisely because the
    rules make it recoverable. So the measure says not only what the notation says but what it expects to be
    worked out, which is the half the rules have to supply.

    It keeps nothing: built once, it is given the sightings on every call."""

    def __init__(
        self, information: Information | None = None, statistics: Statistics | None = None
    ) -> None:
        # The one thing it measures in, given rather than made, so that every learner counting bits
        # counts the same bits.
        self._information = Information() if information is None else information
        # How sure a saying is, counted rather than divided. Whether a share may be believed from two
        # sightings is a question about counting, and it is answered in one place.
        self._statistics = Statistics() if statistics is None else statistics

    def learn(
        self,
        sightings: Sequence[tuple[State, str, Happening]],
        grammar: Grammar,
        telling: float = 0.5,
        least: int = 20,
    ) -> tuple[Coupling, ...]:
        """The couplings those sightings support, each carrying how much it told us and how often it is right.

        `telling` is how many bits a character must carry about a part to count as saying it, and `least` how
        many sightings a shape must have, and a part be present in, to be measured at all. Both are the
        caller's: how much is worth knowing is a fact about what the knowing is for."""
        said = [(grammar.shaped(notation), notation, happening.said(state)) for state, notation, happening in sightings]
        if not said:
            return ()
        parts = sorted({name for _, _, one in said for name in one})
        by_shape: dict[Shape, list[Sighted]] = collections.defaultdict(list)
        for shape, notation, one in said:
            if shape is not None:
                by_shape[shape].append((notation, one))
        found: list[Coupling] = []
        for shape, held in by_shape.items():
            if len(held) < least:
                continue
            found.extend(self._of(shape, held, parts, telling, least))
        unshaped = sum(1 for shape, _, _ in said if shape is None)
        silent = [part for part in parts if not any(one.part == part for one in found)]
        logger.info(
            "Found %d couplings over %d sightings in %d shapes (%d of no known shape); "
            "%d parts nothing says anything about: %s",
            len(found), len(said), len(by_shape), unshaped, len(silent), ", ".join(silent) or "none",
        )
        return tuple(sorted(found, key=lambda one: -one.telling))

    def _of(
        self, shape: Shape, held: Sequence[Sighted], parts: Sequence[str], telling: float, least: int
    ) -> list[Coupling]:
        """What each place of that one shape says."""
        found: list[Coupling] = []
        for at in range(len(shape)):
            for symbol in sorted({notation[at] for notation, _ in held if at < len(notation)}):
                role = Role(((shape, at),))
                standing = [one for one in held if self._saying(one[0], symbol, at)]
                for part in parts:
                    bits = self._told(held, symbol, at, part, least)
                    if bits < telling:
                        continue
                    value = self._mostly(held, symbol, at, part)
                    found.append(
                        Coupling(
                            symbol, part, value, role, bits,
                            self._surely(held, symbol, at, part, value), len(standing),
                        )
                    )
        return found

    def said_alike(self, couplings: Sequence[Coupling]) -> tuple[Coupling, ...]:
        """Those places that say the same things, made one role.

        **Two places are one role when everything measured at them agrees.** The trailing square of a plain move
        and of a taking one say the same thing about where a move lands, and saying it once is what stops the
        number of sayings growing with the number of shapes. Anything less than full agreement is left alone: a
        place that says most of what another says is a different place, and merging it would put a saying where
        it was never measured.

        Pooled by what each was measured on, so a saying seen four times does not outweigh one seen four
        thousand."""
        by_place: dict[tuple[Shape, int], list[Coupling]] = collections.defaultdict(list)
        apart: list[Coupling] = []
        for one in couplings:
            if one.where is None or len(one.where.places) != 1:
                apart.append(one)
                continue
            by_place[one.where.places[0]].append(one)
        alike: dict[frozenset, list[tuple[Shape, int]]] = collections.defaultdict(list)
        for place, held in by_place.items():
            alike[frozenset((one.part, one.symbol, one.value) for one in held)].append(place)
        found = list(apart)
        for saying, places in alike.items():
            role = Role(tuple(sorted(places, key=lambda held: (held[0].sorts, held[1]))))
            for part, symbol, value in sorted(saying, key=lambda held: (held[0], held[1], str(held[2]))):
                pooled = [
                    one
                    for place in places
                    for one in by_place[place]
                    if (one.part, one.symbol, one.value) == (part, symbol, value)
                ]
                found.append(self._pooled(symbol, part, value, role, pooled))
        joined = len(by_place) - len(alike)
        if joined:
            logger.info("Said %d places as %d roles, joining %d of them", len(by_place), len(alike), joined)
        return tuple(sorted(found, key=lambda one: -one.telling))

    def _pooled(
        self, symbol: str, part: str, value: Value, role: Role, held: Sequence[Coupling]
    ) -> Coupling:
        """Those sayings as one, each weighed by what it was measured on."""
        seen = sum(one.seen for one in held)
        if not seen:
            return Coupling(symbol, part, value, role, max(one.telling for one in held), min(one.surely for one in held))
        return Coupling(
            symbol, part, value, role,
            sum(one.telling * one.seen for one in held) / seen,
            sum(one.surely * one.seen for one in held) / seen,
            seen,
        )

    def said_as_one(self, couplings: Sequence[Coupling]) -> tuple[Correspondence, ...]:
        """Those couplings that are one thing, said once.

        A family speaking of the same part in the same role, whose symbols and whose values each fall into an
        order, and which pairs them off without a gap or a crossing, is one correspondence rather than a list of
        sayings. Said as one it covers a symbol it has never met — a ninth row on a bigger board, a letter
        past the edge of this one — which is the whole of what tells a rule from a table.

        A family that does not fall into an order is left alone. `N` naming a knight generalises to nothing
        and should not be made to.

        **A run that is missing its ends is worse than no run at all, which is what makes the measure upstream
        matter here.** A correspondence over `b`..`h` running with 2..8 does not merely omit the a-file: it
        extrapolates the next letter from the wrong base, so the very thing that was meant to reach past the
        board is wrong about the board. Chess's files came out as `b`..`g` and `b`..`h`, and its ranks as
        `3`..`6` and `2`..`6`, for as long as a symbol's bits were averaged over the sightings it is absent from
        — which capped a rare symbol below any threshold whatever it said. Asked where each symbol stands, the
        files come out whole in every role they appear in.
        """
        by_part: dict[tuple[str, Role | None], list[Coupling]] = {}
        for one in couplings:
            by_part.setdefault((one.part, one.where), []).append(one)
        found = []
        for family in by_part.values():
            said = self._ordered(family)
            if said is not None:
                found.append(said)
        if found:
            logger.info(
                "Said %d families of symbols as one thing each: %s",
                len(found), "; ".join(one.readable for one in found),
            )
        return tuple(found)

    def _ordered(self, family: Sequence[Coupling]) -> Correspondence | None:
        """That family as one correspondence, where its symbols and values run in step."""
        if len(family) < 3:
            return None
        if not all(isinstance(one.value, int) and not isinstance(one.value, bool) for one in family):
            return None
        symbols = sorted({one.symbol for one in family})
        values = sorted({one.value for one in family})
        if len(symbols) != len(family) or len(values) != len(family):
            return None
        said = {one.symbol: one.value for one in family}
        telling = sum(one.telling for one in family) / len(family)
        if any(values[number + 1] - values[number] != 1 for number in range(len(values) - 1)):
            return None
        for turned in (False, True):
            paired = list(reversed(values)) if turned else values
            if all(said[symbol] == paired[number] for number, symbol in enumerate(symbols)):
                return Correspondence(family[0].part, tuple(symbols), tuple(values), turned, telling)
        return None

    def _saying(self, notation: str, symbol: str, at: int) -> bool:
        """Whether that notation has the symbol in that place."""
        return at < len(notation) and notation[at] == symbol

    def _told(self, held: Sequence[Sighted], symbol: str, at: int, part: str, least: int) -> float:
        """How many bits that symbol settles about the part **where it stands**.

        What the part cost to say before, less what it costs among the sightings the symbol stands in. A symbol
        that always accompanies one value leaves nothing to say and carries the whole of it; one that
        accompanies the same spread as everything else saves nothing and is not a symbol. That is the guard this
        has always been for, and it is kept.

        **Asked where it stands and not averaged over where it does not, because those are two questions.** The
        measure used to split the sightings in two — the symbol stands, or it does not — and average the spread
        left in each. That is how much *attending to this one character* is worth over a whole corpus, which is
        a fact about the corpus: it is bounded above by the entropy of the split itself, so a symbol standing in
        under a ninth of the sightings cannot reach half a bit however perfectly it speaks. Chess has eight
        files, putting every letter within a hair of that line, and which ones fell under was decided by how
        often they were played. Measured over six thousand moves, `a` and `h` carried 0.49 and 0.46 bits about
        where a move lands while being right nine hundred and ninety-seven times in a thousand — and each of the
        eight sat exactly at its own ceiling, to the last digit, which is what says the number was measuring
        frequency and not meaning. Asked this way all eight settle the file outright, 2.99 bits of the 2.99
        there are.

        **It also stops the measure answering differently about cases that are the same.** The opening letter of
        a taking pawn's notation narrows where the move lands to two files from eight, whichever letter it is.
        Averaged, the six middle ones came out between 0.16 and 0.51 and straddled the threshold, so two were
        kept and four were not for no reason to do with what they say. Asked where they stand, all six come out
        at about 1.96 of 2.95 — the same answer for the same thing.

        **Narrowing is not settling, and `surely` is what tells them apart.** This says a symbol bears on a part;
        it has never said the symbol is always right, and it says so now for more symbols than before. Reading
        and writing both filter on how sure a saying is, which is counted with a case assumed either way so that
        a symbol seen three times is not believed for agreeing three times.

        **Asked only of the sightings that have the part to answer with.** A part that is sometimes missing —
        what was taken, where nothing was — lets a symbol look informative about the part's *value* when all it
        settles is whether there is one at all. Measured with the missing ones counted as an answer, `x` carried
        as many bits about where the taken piece stood as about the taking itself, and was right about the taking
        every time and about the place one time in seven. Whether a part is there at all is already said by
        `takes` and `changes`, which every happening has, so asking each question only where it can be answered
        loses nothing and stops the two from being confused."""
        standing = [one for one in held if part in one[1]]
        if len(standing) < least:
            return 0.0
        whole = collections.Counter(one[part] for _, one in standing)
        where = collections.Counter(
            one[part] for notation, one in standing if self._saying(notation, symbol, at)
        )
        if not where:
            return 0.0
        return self._information.told(whole, where)

    def _surely(
        self, held: Sequence[Sighted], symbol: str, at: int, part: str, value: Value
    ) -> float:
        """How often that saying is right where it applies.

        What a filter needs, and not what bits measure. A symbol can carry a great deal about a part and still be
        wrong one time in five, and a filter that does not know the difference throws away the reading it was
        looking for.

        Asked of every sighting the symbol stands in, including those with no such part — unlike the bits, and
        deliberately. This is how the saying will be used: a happening missing the part fails the check, so a
        symbol whose part is usually absent is usually wrong, and should be measured as such.

        **Counted with a case assumed either way, so that nothing is certain from nothing.** A bare share says a
        symbol seen twice and right twice is right always, and says it with the same face as one right three
        thousand nine hundred and ninety-nine times in four thousand. A filter reading those alike lets the
        first through on no evidence at all — and being filtered on is the whole of what this number is for."""
        counted = [one[1].get(part) for one in held if self._saying(one[0], symbol, at)]
        if not counted:
            return 0.0
        return self._statistics.share(sum(1 for one in counted if one == value), len(counted))

    def _mostly(self, held: Sequence[Sighted], symbol: str, at: int, part: str) -> Value | None:
        """What that part mostly is where the symbol stands — what the symbol is taken to say.

        Mostly and not always: a symbol that says a thing nine times in ten is saying it, and the tenth is either
        a second meaning this has not told apart yet or a role it has not been given.

        Asked of the same sightings the bits were asked of, and for the same reason. Were the missing ones
        counted here while the bits ignored them, a saying could be measured on the happenings that have the part
        and then report what the ones without it mostly are, which is nothing."""
        counted = collections.Counter(
            one[1][part] for one in held if part in one[1] and self._saying(one[0], symbol, at)
        )
        return counted.most_common(1)[0][0] if counted else None
