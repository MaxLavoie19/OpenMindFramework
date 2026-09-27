# Interfaces: sniping one let-through

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
is a question, not a decision.

Adds a second way of learning a constraint beside the one there is. Nothing about the readings, the guard, the
price or the distiller changes, and `RefusalLearner.learn` is untouched.

## The finding it answers

Measured tonight, on a run's own constraints against a position twelve moves in:

```
                     candidates   legal among them
  all                     14400                 43
  left standing             118                 43
```

Forty-eight constraints close ninety-nine per cent of the space and **not one of them refuses a legal move**.
Every error is the same error: seventy-five candidates the rules allow and the game refuses. Yet the learner
spends its sixty seconds a position on whatever cases are unaccounted for, shallowly, and produces constraints
of one or two conditions — while the residue concentrates on the long-range pieces, where thirty-one of the
seventy-five are a rook moving in a way no rook moves, and saying so takes three conditions.

Measured separately, on eight cases a mature set still left standing, against a guard of eighteen hundred legal
moves: **half lost between a third and a half of their coverage** to the search stopping at the first size that
worked. The deeper search was then run as a whole arm for twelve positions and did not pay — because at twelve
positions the easy refusals still dominate and one condition genuinely is the right answer. It is on the
residue that depth is worth paying for, and nothing today spends its time there.

## What it is

One candidate the rules allow and the game refuses, worked on by itself for as long as it takes, with what it
has tried kept so the next attempt carries on rather than starting again.

**Reentrant, because "as long as it takes" cannot mean one sitting.** A search that must finish before anything
else happens is a search that cannot be given five minutes. So a pursuit is a thing that can be put down and
picked up: it carries the size it has reached, how far into that size it got, and what it has spent. A
scheduler then gives it whatever time there is — this process's spare seconds, or a worker where one is free.

**Ranked by how much else it fixes.** A body that refuses this candidate and forty others spread over ten
boards is a rule about the game. One that refuses this candidate and forty others on the same board may be a
rule about that board. So candidates are pooled across positions and a body is scored against the pool.

## It ends properly, not on a clock

A body is worth asking the guard about only where **every part of it one condition shorter turned away a legal
move** — anything containing a part that already passed refuses no more than that part and costs more to say.
That prune is already in `RefusalLearner.worth_trying` and `HypothesisTester.tried`.

It gives the search a real end. If nothing slipped at size *n*, then every body of size *n* either stood or was
dominated, so **every** body of size *n+1* contains a part that stood, and none of them can beat what is
already standing. The search is finished, at any depth, without a ceiling and without a timeout.

That matters twice. A pursuit that ends with something standing has found the shortest-and-cheapest thing there
is. A pursuit that ends with **nothing** standing has shown something much stronger than "we ran out of time":
*no combination of the readings distinguishes this candidate from the legal moves*. That is a missing reading,
and it is a discovery to be reported rather than a failure to be retried — the same thing `unexplained` says at
the position level, said here about one candidate with a proof behind it.

## The interfaces

### `inference/model/pursuit.py` — `Pursuit`

Frozen, slotted, and carries no services: it is what a scheduler keeps, hands to a worker, and gets back.

```python
@dataclass(frozen=True, slots=True)
class Pursuit:
    """One candidate the rules allow and the game refuses, and how far the search for what refuses it has got."""

    case: Example
    #: Its readings, tied once, so resuming does not read the position again and two sessions offer the
    #: same bodies in the same order.
    offered: tuple[Literal, ...]
    #: The size being searched. 0 before it starts.
    size: int = 0
    #: How far into that size's combinations it has got, as a count — the `start` a range is resumed from.
    at: int = 0
    #: The best body found so far, by the price, or None while nothing stands.
    found: Clause | None = None
    #: Seconds spent on it across every session.
    seconds: float = 0.0
    #: Why it stopped, where it has: nothing slipped one size shorter, so no deeper body can beat what stands.
    exhausted: bool = False
```

**Open:** whether `offered` belongs here or is read back from the case each time. Carrying it makes a pursuit
big to send to a worker; re-reading it risks two sessions offering bodies in a different order, which would
make `at` mean different things. Carrying it is the safe answer and the one written above.

### `inference/service/sniper.py` — `Sniper`

Stateless, built once, injected. It holds the learner it borrows the guard and the price from, and nothing else.

```python
class Sniper:
    def __init__(self, refusal_learner: RefusalLearner, clock: Callable[[], float] = time.monotonic) -> None: ...

    def pursue(
        self,
        pursuit: Pursuit,
        guard: CaseIndex,
        table: HypothesisTable,
        pool: Sequence[Example],
        seconds: float,
    ) -> Pursuit:
        """That pursuit carried on for that many seconds, and where it got to.

        It resumes at the size and offset the pursuit carries, prunes each size past the first by what slipped
        one shorter, writes everything tried into the table, and comes back whether or not it found anything.
        Where a size is finished and nothing slipped in it, the pursuit comes back exhausted."""

    def ranked(self, standing: Sequence[Clause], pool: Sequence[Example]) -> tuple[tuple[Clause, int], ...]:
        """Each body with how many of the pool it also refuses, most first, ties broken by what it costs.

        The pool is candidates the rules allow and the game refuses, gathered across positions. A body
        refusing many of them over many boards is a rule about the game."""

    def started(self, case: Example, readings: CandidateReadings) -> Pursuit:
        """A pursuit of that candidate, its readings tied once."""
```

`HypothesisTable` is the resumable state and needs no change: it is already keyed by body, already keeps what
slipped so it is not retried, and `slipping(size)` already hands back what the next size prunes by. Handing a
slice of one size to a worker needs no change either — `HypothesisTester.tried` already takes `start` and
`stop` over a fixed combination order, and says in its own docstring that a range is "a piece of work that can
be handed anywhere without handing over the work itself".

### Not in this proposal

**The scheduler.** Which pursuit to work on, for how long, and whether to spend a worker on it. It wants its
own proposal and it wants this measured first. Until it exists, a caller with a list of pursuits and a loop is
enough to find out whether sniping works at all.

**Gathering the pool.** A candidate the rules allow and the game refuses is found by asking the rules about
every candidate — fourteen thousand a position, about two minutes at the current constraint count. Affordable
for something that spends minutes anyway, but it is the real cost and it is the scheduler's to pay. The scouts
already ask the constraints about every position they walk and would be the cheap place to gather it.

## Verification

| Green when |
|---|
| A pursuit given a case a one-condition body refuses comes back with it, exhausted, in one session |
| A pursuit given a case needing three conditions, and one second at a time, reaches it across sessions and reaches the **same** body it reaches in one |
| A pursuit whose readings cannot tell its case from the legal moves comes back exhausted with `found` None |
| Two bodies refusing the same candidate are ranked by how many others in the pool they refuse, not by which was found first |
| A pursuit resumed against a table another pursuit filled does not retry what that one tried |

On chess, the claim to measure: **the residue falls.** Take the seventy-five candidates a run's own constraints
leave standing, snipe them, and count how many are still standing with the new constraints added — against the
same count from giving the ordinary learner the same total seconds.

## Open questions

- **Whether a pursuit stops at the first body that stands or finishes its size.** Finishing gives the ranking
  something to choose between; stopping is faster. Written above as finishing the size, because a sniper's
  whole point is that it is not in a hurry.
- **What a pool is scored against when it changes.** Constraints found by one pursuit refuse some of the pool,
  so the pool shrinks under the pursuits. Rescoring every pursuit each time is the honest answer and may be
  dear; scoring against the pool as it was is cheap and slowly wrong.
- **Whether an exhausted pursuit with nothing found should propose a reading.** It has shown that the
  vocabulary cannot tell its case apart, which is exactly the evidence for wanting a new one — and adding a
  reading is a thing that stops and asks. So: report, and ask.
