# Interfaces: one table of hypotheses, many workers

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open**
isn't decided.

## What made this necessary

Scouting works and immediately overran the thing it feeds. Measured on the first run, after **one** position:

```
    walks   moves handed over   positions waiting   positions learned from
      846             33,840               2,538                        1
```

The parent never reached position two. Walking is about three hundred positions a second; learning from one is
about one per ten. Throttling the walkers is the wrong end of it — that idles twenty-three cores to protect one.

The first answer was one learner per scouted position, merging at the end. It parallelises the *data*. The
better answer parallelises the *hypotheses*, because the work each learner does is mostly the work every other
learner is also doing: they all re-derive the same short bodies against their own board, and none of them can
use what another has already ruled out.

## What a node is, and why sharing is sound

A node is a **body** — a conjunction of conditions, the head always `refused`. Its children are that body with
one more condition, drawn from the readings a case carried. `_briefest` already walks this tree by increasing
size; it walks it alone, per position, from nothing, every time.

**Conditions only ever narrow.** So a body that fails to refuse the case it was grown for can never be fixed by
extending it, and its whole subtree is dead — one worker finding that spares every other worker. This is
Popper's generate–test–constrain, listed in [learner-plan.md](../learner-plan.md) as deferred "until the by-size
search proves slow". It is slow.

**Slipping does not prune.** A body that turns away a move the game allows may be saved by another condition, so
a failure against the guard rules out that body and nothing below it. The two failures are not the same and are
not recorded the same way.

## Why a table rather than a frontier

What is worth sharing is not which bodies to try next — it is **what each body refuses**. A body's coverage over
the pooled cases does not depend on which rule you were looking for when you wrote it down, so it is computed
once and serves every case it covers. `RefusalLearner.coverage` is already this idea, per learner, thrown away
each position.

Two things follow that a frontier would not give:

- **Corroboration finally has something to work with.** `--corroborated` asks how many distinct boards a
  constraint was grown from, and the pool it counts over has always been one position's cases plus a sample.
  Here every body is tested against every pooled board by construction.
- **The open question in the plan becomes stateable.** "What determines how many constraints the learner settles
  on" is unanswered because description length is never applied to the *set* — a repair heuristic sets the
  count. Given bodies and what each refuses, choosing the set is a covering problem that can be priced by
  description length instead of by policy. This does not answer it; it makes it askable.

## Models

**`inference.model.hypothesis.Hypothesis`** — one body and what it came to.

```python
clause: Clause            # the head is always `refused`; the body is what varies
refusing: frozenset[int]  # which pooled cases it refuses, by their place in the pool
slips: int                # how many of the guard it turns away — nought is what makes it usable
```

`refusing` is places and not cases, because places are integers and cases are a hundred and fifty readings
each. Nothing about a case crosses between processes.

**Open.** Whether `Hypothesis` carries its size or the caller takes `len(clause.body)`. Deriving it is free;
carrying it makes the table sortable without touching the clause.

## The pool, and the one thing that is expensive

A worker has to have the cases to answer about. Sending them is out — a position is 14,400 cases of 152
readings. So a worker is sent **positions** and reads them itself, exactly as a scout does.

But reading a position is about ten seconds, and a pool of ten positions is a hundred. Paid per batch of
hypotheses that is ruinous; paid once per worker it is a startup cost.

**So a worker keeps the pool it read, keyed by which positions it is.** This is the one piece of state in an
otherwise stateless design, and it is a cache rather than a memory: given the same pool it answers from what it
has, given a different one it reads again.

**The budget this needs.** A pooled position costs roughly 175 MB in each worker that holds it. Ten positions
across twenty-three workers is about 4 GB against cinamon's 62 GB, which is affordable and is not nothing. How
many positions the pool holds is the caller's number, like `--remembered`.

**Open.** Whether the pool is the scouts' best positions, a sample of everything they saw, or both. The first is
where the rules are most wrong; the second is what keeps a rule from fitting the places rules are wrong.

## Services

**`inference.service.hypothesis_tester`** — given bodies and a pool, what each refuses. This is the worker
function, and it is the only thing that runs in the other processes.

**`inference.service.hypothesis_table`** — holds what has come back, grows the frontier from it, and hands out
the next batch. Lives in the parent. It is the only part that knows about dead subtrees, because knowing that a
body covered nothing is the same fact as its `refusing` being empty.

**Open.** Whether assembling rules from the table belongs here or stays in `RefusalLearner`. Against a new home:
the greedy covering, the guard and the repair are all there already. For: what this replaces is precisely the
search half of that service, and leaving the two entangled is what made the search unshareable.

## How it runs

`TaskRunner.stream`, as scouting does: endless calls, each worker handed a batch when it comes free, the batch
chosen *then* from the current frontier. Results fold into the table as they arrive. The parent's own thread
assembles rules and distils.

## What this replaces, and what it keeps

**Keeps** — the scouts, `Scouted`, `Sighting`, `CandidateReadings.of_allowed`, `RefusalLearner.wrongly`. All of
them are about finding positions and judging them cheaply, which is unchanged.

**Replaces** — `position_worker.learned_from_a_position`, written an hour ago: one learner per scouted position,
merging at the end. It is the data-parallel design and the weaker one, for the reason above.

**Leaves alone** — distilling, repair, the hypothetical, and everything about what a reading is. This is about
who searches and what they share, not about what can be said.
