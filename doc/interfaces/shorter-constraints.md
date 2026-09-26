# Interfaces: constraints that start short

**Superseded by [moves-as-offsets.md](moves-as-offsets.md). Built, measured, reverted.** Growing a constraint
from its case does shorten it — to one condition, refusing nothing legal — and every constraint it produces is
a coordinate, because within a single position a coordinate separates maximally and what stands on a square
separates weakly. `refused :- destination(1, 1)` is short and generalises to nothing. Length was the symptom.
The disease is that one flat set of constraints has to tell a bishop's moves from a knight's with no condition
naming which is which, so it reaches for the board or for the coordinates, and this proposal only chose which.

Kept for the measurements in it, which stand.

Changes where a constraint comes from. Nothing about the vocabulary, the readings or the distiller's job
changes.

## The finding

Measured, after being guessed at wrongly twice — once as covered cases never generalising what covers them, once
as widenings refused for reaching a legal move. Both are false, and the numbers say so:

```
widening the long ones against 4075 refused cases of a position they have never seen

 from   tried  no widening  reaches a legal move   kept  shortest kept
  162    4075            0                  1930   2145            158
  161    4075            0                  2052   2023            157
  161    4075            0                  3587    488            158
```

**Generalising does not shorten.** No case fails to yield a widening, about half the widenings are accepted, and
an accepted one takes a hundred and sixty-one conditions to a hundred and fifty-seven. `generalised` keeps a
condition wherever it can pair it with a reading of the same predicate and drops it only where no partner
exists — and since both the clause and the case describe a whole board, every `holds` and `grid` condition
always finds one. The least general generalisation of two board-sized clauses is another board-sized clause with
more variables in it.

**So constraints are born board-sized and stay that way.** `_outright` says everything read of a case, which on
chess is about a hundred and sixty conditions, and its docstring defends this: "What is genuinely surplus comes
out later, when the constraints are distilled." For some it does — the store holds fifty constraints of nine
conditions or fewer, among them `refused :- turn(X4), holds(destination row, destination column, grid,
square(X6, piece(X4, X7)))`, which is moving onto your own piece, found with no chess vocabulary. For most it
does not: ninety of a hundred and forty are eighty conditions or more, and twenty-eight are around a hundred and
fifty-eight, which is two boards folded together.

**And one position is enough to be believed.** `_supported` asks for `positions` distinct positions and defaults
to one, so a constraint whose conditions are load-bearing for a single board is kept. That is the postcode, and
the gate meant to catch it is open.

## Why it matters beyond tidiness

Asking whether a constraint covers a candidate walks its body, so a hundred and sixty conditions against four
thousand candidates against a hundred and eighty constraints is most of a position's budget. The cost of
everything else in the loop is set by this.

## What changes

### A constraint is grown from a case, not said outright

Today: say the whole case, then widen it and hope distilling finds the surplus. Proposed: say nothing, and add
conditions from the case's own readings — at each step the one that most reduces what the clause reaches among
the moves the game allows — until it reaches none.

What comes out is the shortest thing sayable from that case that refuses it without reaching a legal move. Short
by construction, rather than short if the distiller can prove every condition surplus later.

```python
def _grown(self, example: Example, index: CaseIndex) -> Clause | None:
    """That case said as briefly as it can be while reaching nothing the game allows.

    The dual of `repaired`, which adds conditions to a constraint that reaches too far. This starts from
    nothing and adds the same way, so there is one idea in two places rather than two."""
```

**The objection this used to face does not apply.** `_widened` records it: "Growing downward, every candidate
condition is scored against every case, and there are thousands of candidates." True of growing from the whole
vocabulary. The candidates here are one case's readings — about a hundred and sixty — and `repaired` already
scores exactly this way, so the machinery and its cost are both known.

### Generalising then stays short

Anti-unification of two short clauses is short. The length never has to be recovered because it is never lost,
and `generalised` itself does not change.

### A constraint has to recur before it is believed

`_supported`'s `positions` should default to more than one, and the chess entry point should pass what it wants.
A rule claimed of a game should turn up in more than the position it was read from; a coincidence has to recur
somewhere else to survive, which is what the parameter was written for and what defaulting it to one gives away.

## What goes

`_outright`, and the part of its docstring arguing that saying everything is safe because distilling will sort
it out. The measurement above is the counter-evidence and belongs in its place.

## How it is judged

Body-length distribution and constraints-per-position, before and after, over the same positions — reported as
counts with a header, not as a claim about quality. The two error counts must not get worse: nothing wrongly
refused is the column that has held all along.

## Open

- **Whether growing greedily strands a constraint.** Picking the condition that releases most may block a
  shorter body reachable by picking another. The distiller still runs afterwards, so a stranded clause is not
  permanent, but nothing here searches.
- **What `positions` should default to.** Two makes a constraint recur once. Higher is stricter and slower to
  learn anything at all, since a rule seen in one position waits for its second. Not decided, and it is the
  caller's number in any case.
- **Short and wrong is still possible.** `refused :- destination(1, X2)` says nothing may move to row one, and
  survives because nothing has contradicted it yet. Shortness is not generality, and this changes nothing about
  that.
