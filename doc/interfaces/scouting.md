# Interfaces: scouting positions worth learning from

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

## The measurement this comes from

Four positions, profiled 2026-09-24, before any change:

```
 seconds   share  what                                    (nested entries overlap their parent)
   245.6    100%  the whole run, four positions
   116.2     47%  _matching — the dashboard's count of hand-written rules found
   106.0     43%  reading the 14,400 candidates        (10 calls for 4 positions)
    64.9     26%    └ of which, inside scored          (6 of those 10 calls)
    22.5      9%  repairing constraints that refuse a legal move
    17.3      7%  refusal_learner.learn  ← the actual learning
    14.1      6%  coverage
     8.2      3%  distilling
     1.6      1%  choosing where to go next
     0.1      0%  the predictor and the notation, together
```

Two things follow, and the second is this document.

**The predictor and the decoder are not slow; they are starved.** Together they are a tenth of a second against
two hundred and forty-five. They are not waiting on cores — they are waiting on *moves*, and the constraint
learner hands them one every thirty-one seconds. The decoder learns nothing at all until twenty notations have
been seen, which is currently ten minutes of wall clock. The predictor's standing complaint in
[learner-plan.md](../learner-plan.md) — "overfitted on thin evidence, like everything else" — is a sampling
problem wearing an algorithm's clothes.

**Judging a position is four thousand times cheaper than learning from one.** Choosing where to go next
evaluates about thirty boards in 1.6 seconds over four positions — thirteen milliseconds to read a position's
legal moves and ask the constraints about them. Learning from a position costs thirty-one seconds, because it
reads all 14,400 candidates and searches. One core judges seventy-five positions a second; the learner consumes
one per half-minute.

So the walk should not be driven by the learner. A cheap process can look at tens of thousands of positions and
hand over the few worth the expensive one's time.

## What a scout is

A worker that walks a game of its own and does only the cheap things:

- **Gathers sightings.** Every move it plays, with the position before it, what the game says it did, and what
  the game called it. That is the whole diet of the predictor and the decoder, and the scouts are where it
  now comes from.
- **Scores each position** by how many actions the game allows the constraints refuse there. That is the mistake
  nothing else tells OMF about — a move it will never make and will never hear was possible — and it is what
  `_next` already maximises, one ply at a time, in the expensive process.

It never reads the full candidate space, never learns a constraint, and never distils. Everything it hands back
is small: a board, thirty-odd actions, a handful of changes, a string.

Games of their own, one seed each, as `learned_from_a_game` already does. A position is one move from the last
and differs in two squares, so handing out neighbouring positions would have every worker looking at nearly the
same thing.

## Models

**`inference.model.scouted.Scouted`** — one position a scout thought worth the learner's time.

```python
where: Evidence   # the position and the actions the game allows there: all the learner needs
wrongly: int      # how many of those actions the constraints as they stand refuse
```

`Evidence` is already exactly the parcel: a `State`, the action's name, and the legal actions. The 14,400
candidates are *derived* from it, in the parent, so they never cross a process boundary.

**`predictor.model.sighting.Sighting`** — a move seen, and what it was called.

```python
watched: Watched   # the position before, the action, and the changes
said: str          # the game's own notation for it
```

Today these are two parallel lists in `learn_constraints.py` kept in step by hand. Crossing a process boundary
is where a pairing kept by convention becomes a pairing kept by a type.

**Open.** Whether `said` belongs on `Watched` instead. Against: a game need not have a notation, and `Watched`
is what the predictor is given. For: two things that must stay aligned are safest in one object.

## Services

**`CandidateReadings.of_allowed(evidence) -> tuple[Example, ...]`** — the cases for the actions the game lists,
and no others, sharing the position's own readings between them.

This is the scout's primitive and it already exists, written out by hand in `_next` with two `# noqa: SLF001`
comments reaching into `_of_parameters` and `_aliases`. Naming it makes the cheap read a thing the codebase
offers rather than a thing one caller knows how to do.

**`inference.service.scout.Scout`** — stateless, given a learner and readings.

```python
def scored(self, clauses, evidence) -> int:
    """How many actions the game allows those constraints refuse there."""
```

**Open.** Whether `Scout` earns a service of its own, or whether this is one method on `RefusalLearner` beside
`scored`. It is three lines. Against a new service: it is three lines. For: `RefusalLearner` is already the
largest service in the codebase.

The *walking* stays in the game — chess pushes chess moves — so the worker function lives beside
`learned_from_a_game` in `openmind_chess`, and returns `(tuple[Scouted, ...], tuple[Sighting, ...])`.

## How the parent uses them

`TaskRunner.stream` with `count=None`, on a thread of its own in the parent:

- Each call's arguments are chosen when a worker takes it, which is what `arguments_for` is for — so every scout
  starts with the constraints **as they stand**, and scouting stays pointed at what is currently wrong.
- `on_result` folds each worker's sightings and scouted positions into what the parent holds, under a lock.
- The parent's own thread does the expensive thing: take the best-scoring position off the queue, learn from it,
  distil, repair. The predictor and the decoder learn from the sightings as they arrive, since they are free.

The GIL is not in the way: the stream's thread spends its life blocked in `wait()` on the workers' pipes.

**Open.** What the parent does when the queue is empty — wait for a scout, or learn from where it already is.
**Open.** Whether a scouted position is stale after the constraints change under it, and if so how stale is too
stale. A position scored against constraints from a minute ago may no longer be where they are most wrong.

## What this does not do

It does not make `learn` faster. Learning is 7% of the run and it is the part that is supposed to cost
something. What it makes faster is everything around it — and it makes the other two learners a hundredfold
better fed, which is a different kind of faster and probably the more valuable one.

It also leaves `_matching` alone, at 47% of the run. That is the dashboard's count of how many hand-written
rules have been found, and it re-asks forty hand-written constraints about all 14,400 candidates every position.
It is a measurement, not learning, and it wants its own decision: parallelise it, compute it from the coverage
already in hand, or take it every tenth position rather than every one.

## Already done, not waiting on this

Reading the same 14,400 candidates two and three times a position. `RefusalLearner.scored` now takes the cases
where the caller holds them; `learn_constraints.py` hands over the ones it just learned from. Measured over the
same four positions: position 1 from 22.1 to 11.3 seconds, position 2 from 29.6 to 18.6.
