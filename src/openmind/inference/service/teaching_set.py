import logging
from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


class TeachingSet:
    """The fewest positions that let a learner test every rule it is being handed.

    **A rule sent without a position to test it on is a rule that has to be believed.** Handing over what was
    learned is cheap and exact where both sides share a vocabulary, and what arrives is the teacher's
    mistakes along with its insights. That is survivable — a mistake handed over is a hypothesis, and the
    learner has the same game to put it to — but only where the learner is given somewhere to put it. A rule
    that fires nowhere in what it was sent cannot be checked, however wrong it is.

    So the positions to send are not the teacher's whole history and not a sample of it. They are the ones
    that make each rule *answerable*: a position where the rule fires is a position where the game will say
    whether it was right.

    **Fewest, because a teacher that sends everything has taught nothing about what matters.** This is the
    shape the machine-teaching literature calls a teaching set, and the greedy cover is the usual
    approximation — each position taken is the one testing the most rules still short of their quota, which
    is within a log factor of the smallest such set and needs nothing the teacher does not already have.

    **A rule nothing can test is reported rather than dropped.** It may be right; it may be the overfitted
    coincidence that has been believed for weeks because no position anybody visited disagreed with it.
    Either way the honest thing is to hand it over saying so, not to hide it or to pretend a position tests
    it."""

    def covering(
        self, firing: Mapping[str, Sequence[int]], least: int = 1
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        """Which positions to send, and which rules nothing sent can test.

        `firing` is, per rule, the positions it fires in — the ones where the game will say whether it was
        right. `least` is how many tests each rule should get: one shows whether it is ever wrong, more shows
        whether it is wrong often, and which of those is wanted is the teacher's business and not this one's.

        The positions come back in the order they were chosen, most useful first, so a teacher with less room
        than the whole set can send a prefix and still have spent it on the rules least covered."""
        wanted = {rule: min(least, len(where)) for rule, where in firing.items() if where}
        untestable = tuple(sorted(rule for rule, where in firing.items() if not where))
        chosen: list[int] = []
        taken: set[int] = set()
        short = dict(wanted)
        while short:
            counts: dict[int, int] = {}
            for rule, owing in short.items():
                if owing <= 0:
                    continue
                for at in firing[rule]:
                    if at not in taken:
                        counts[at] = counts.get(at, 0) + 1
            if not counts:
                break
            # The position testing the most rules still short, ties to the earliest so a run is repeatable.
            best = min(counts, key=lambda at: (-counts[at], at))
            chosen.append(best)
            taken.add(best)
            for rule in list(short):
                if best in firing[rule]:
                    short[rule] -= 1
                if short[rule] <= 0:
                    del short[rule]
        logger.info(
            "Teaching %d rules with %d positions, %d of them testable %d times over; %d nothing can test",
            len(firing),
            len(chosen),
            len(wanted),
            least,
            len(untestable),
        )
        return tuple(chosen), untestable

    def tested(self, firing: Mapping[str, Sequence[int]], sent: Sequence[int]) -> dict[str, int]:
        """How many times each rule can be tested by what was sent.

        For reading a teaching set back: a rule at nought was handed over untestable, and a teacher sending
        a prefix of the chosen positions needs to know which rules that cost it."""
        held = set(sent)
        return {rule: sum(1 for at in where if at in held) for rule, where in firing.items()}
