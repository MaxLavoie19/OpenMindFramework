import hashlib
import logging
from collections.abc import Sequence

from openmind.inference.model.rule_set import RuleSet
from openmind.statement.model.clause import Clause

logger = logging.getLogger(__name__)


class RuleLibrary:
    """Every rule once, however many sets point at it.

    **What a rule costs is paid once, and what a set costs is its pointers.** A rule serving six movers is a
    sixth as dear per use as six copies of it, which is the first pressure toward generality that does not have
    to recognise generality first. A rule naming a particular square can only ever serve one mover in one
    position, so it never gains a second pointer and goes on costing full price — nothing has to detect that it
    is a postcode, it simply never gets cheaper.

    **Sameness is the rule's own.** Two sets arriving at the same rule arrive at the same pointer, because the
    pointer is made from what the rule says. So sharing needs no comparing pass and cannot be missed: a rule
    learned again is a rule recognised.

    **What nothing points at is forgotten.** Not tidying-up — it is the other half of the pricing. A rule that no
    set needs any more stops being paid for, so a theory that has been improved is cheaper than one that has
    merely been added to, and repairing a set actually makes the whole thing smaller.

    It keeps the rules and who points at them, and nothing else: no history, no scores, no order."""

    def __init__(self) -> None:
        self._rules: dict[str, Clause] = {}
        self._sets: dict[object, RuleSet] = {}

    def put(self, clause: Clause) -> str:
        """That rule in the library, and what to call it.

        The same rule put twice is one rule and the same name both times, which is what makes two sets arriving
        at it separately share it without either knowing about the other."""
        pointer = self._named(clause)
        self._rules.setdefault(pointer, clause)
        return pointer

    def at(self, pointer: str) -> Clause | None:
        """The rule of that name, or nothing where it has been forgotten."""
        return self._rules.get(pointer)

    def of(self, what: object) -> RuleSet:
        """The set of rules for that kind of thing, empty where it has none yet."""
        return self._sets.get(what, RuleSet(what))

    def told(self, what: object, clause: Clause) -> str:
        """That kind of thing also refuses by that rule. The rule goes in the library, the set points at it."""
        pointer = self.put(clause)
        self._sets[what] = self.of(what).with_rule(pointer)
        return pointer

    def untold(self, what: object, pointer: str) -> None:
        """That kind of thing no longer refuses by that rule, and the rule goes where nothing points at it."""
        self._sets[what] = self.of(what).without(pointer)
        self.forgotten()

    def rules(self, what: object) -> tuple[Clause, ...]:
        """The rules of that kind of thing, as rules rather than as pointers."""
        return tuple(
            held for held in (self.at(one) for one in self.of(what).rules) if held is not None
        )

    def pointed_at(self, pointer: str) -> int:
        """How many sets name that rule.

        Which is how widely it holds, read rather than declared: a rule every set names is a rule of the game,
        and one a single set names is that mover's own."""
        return sum(1 for one in self._sets.values() if pointer in one)

    def forgotten(self) -> tuple[str, ...]:
        """Those rules nothing points at any more, removed, and their names."""
        gone = tuple(one for one in self._rules if not self.pointed_at(one))
        for one in gone:
            del self._rules[one]
        if gone:
            logger.info("Forgot %d rules nothing points at any more", len(gone))
        return gone

    def shared(self) -> tuple[tuple[str, int], ...]:
        """Each rule and how many sets point at it, the most widely held first.

        What this is for is saying where the boundary between a rule of the game and a rule of one mover fell,
        without anybody having drawn it."""
        return tuple(
            sorted(((one, self.pointed_at(one)) for one in self._rules), key=lambda held: -held[1])
        )

    @property
    def sets(self) -> tuple[RuleSet, ...]:
        return tuple(self._sets.values())

    def all(self) -> tuple[Clause, ...]:
        """Every rule in the library, whoever points at it."""
        return tuple(self._rules.values())

    def cost(self, per_rule: float = 1.0, per_pointer: float = 1.0, exponent: float = 2.0) -> float:
        """What the whole thing costs to say: each rule once, plus what it takes to point at them.

        The rules are charged as they would be charged anywhere — a body costing more than in proportion to its
        length, so two simple rules beat one long one. What is new is that they are charged *once*, so a set
        naming a rule somebody else already needed adds a pointer and no rule at all."""
        rules = sum(per_rule + float(len(one.body)) ** exponent for one in self._rules.values())
        return rules + per_pointer * sum(len(one) for one in self._sets.values())

    def _named(self, clause: Clause) -> str:
        """What to call that rule: a name made from what it says, so the same rule is always the same name.

        Made from the rule rather than counted out, because two sets that learn the same thing separately have
        to arrive at one pointer without either having seen the other's."""
        said = clause.readable.encode()
        return hashlib.blake2b(said, digest_size=8).hexdigest()

    def measured(self) -> str:
        """How the library stands, for a log: how many rules, how many sets, and how much is shared."""
        shared = [count for _, count in self.shared()]
        held = sum(1 for one in shared if one > 1)
        return (
            f"{len(self._rules)} rules over {len(self._sets)} sets, {held} of them shared, "
            f"costing {self.cost():.0f}"
        )
