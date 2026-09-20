# Interfaces: the inference bootstrap

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

The first step of bootstrapping a game is to let OMF **ponder reasonable heuristics**: from the rules and the legal
actions alone, before a single game is played, propose what a position might be worth and why. The second step is
self-play, where the agent tries what it deduced, finds out whether it holds up in a game, and meets positions worth
studying.

**Nothing here knows any chess.** No piece values, no tables, no formulas: OMF derives what it uses by experimenting
on the game it was given, and the same deriving runs on tic-tac-toe, on a game nobody has written yet, and on chess.

## The instruments it already has

| What | What it does |
|---|---|
| the CSP | how many legal actions a player has in a position — the only measuring stick that needs no chess |
| `Mechanics.with_value(state, model, at, value)` | a position with one cell changed: what a removal experiment is made of |
| `GameRelaxer.relax(context, relaxation)` | the game with rules dropped, which chess already declares as `teleport` and `ignoring check` |
| `ExpressionSearch`, `ExpressionGenerator` | composes candidate rules out of the readings a position offers |
| `SparseFitter` | keeps the terms that earn their place once there are games to fit on |

## Tools, not deductions

The line this step has to hold: OMF is given **tools and a vocabulary**, and makes the deductions itself. A tool is
something it can ask of any game. A deduction is what it concludes about this one.

| Tool | The question it answers |
|---|---|
| the CSP | how many legal actions a player has here |
| a counterfactual: `Mechanics.with_value(state, model, at, value)` then ask again | what changes if this part of the state were different — including empty, since a cell holding `None` holds nothing |
| `GameRelaxer.relax(context, relaxation)` | the same game with rules dropped, which is easier to win and so says how far a win is |
| `Mechanics` readings | what a position offers to be read: what each model holds, how much of it, what the rules single out |
| `ExpressionSearch`, `ExpressionGenerator` | every candidate expression those readings compose |
| `SparseFitter` | which candidates earn their place against a target |

**None of these is a heuristic.** "A piece is worth what it reaches" is not written anywhere: it is one of the things
the search can compose, out of a counterfactual reading and a count, and it survives only if it fits the target
better than the alternatives. On a game where holding things is worthless, it won't.

## What it fits against, before any game is played

A fit needs a target, and self-play hasn't happened yet. What OMF has without games:

- **What the game pays.** Positions it can reach that are over carry their payoffs, which are the only values known
  for certain.
- **A relaxed game's distance to a win.** Relaxing until the game is small enough to read out gives a position a
  value that cost no self-play — that is what relaxations are for, and chess already declares two.
- **What a deduction proves.** A position proven won is worth the win, as the position deducer already reads.

These are what a candidate expression is measured against. Which of them to spend the budget on is OMF's, and so is
finding out that one of them was a waste.

**A relaxation can be a bad one, and that is OMF's to discover.** The relaxer drops a constraint, which widens what
is legal: chess's teleport turns twenty moves into about a thousand, so the relaxed game is harder to read out than
the real one, not easier. Nothing here decides in advance which relaxations are worth it. The ponderer tries what the
game offers, and a relaxation that couldn't be solved inside its budget, or whose values never varied, or whose
values fitted nothing, is recorded as what it was: a way of bootstrapping that didn't pay here. That record is what
the bootstrapper — a model of its own, later — learns from, so a second game doesn't pay for the same lesson.

If every relaxation a game offers turns out to be a bad one, that is a finding about the relaxer: it can only loosen
a game, and a bootstrap wants relaxations that shrink it. A tool to sharpen when the evidence says so, not a guess to
build on now.

## The ponderer

```python
class HeuristicPonderer:
    """From the rules alone: gathers positions it can value without playing, composes the candidates its readings
    allow, fits them, and keeps what held up — the heuristics worth trying in this game."""

    def ponder(self, knowledge_base, game, budget: Budget) -> tuple[RuleRecord, ...]: ...
```

What it gives back is declared as an **open** ruleset in the game's context and registered as a model of the position
value task, so the search can plan with it at once and the training step can fit better weights on what self-play
brings. Pondering is a task OMF spends time on, so it takes a budget like everything else.

## Seeing whether it got there by itself, and used it

Tests can show that OMF **can** reach a heuristic: given the readings and a target, the fit finds it. That is not the
same as OMF reaching it **by itself** from a game it was handed, nor the same as the heuristic being any use. Three
different things, and this step has to show all three.

| What | How it is shown |
|---|---|
| **It can** | tests: the fit finds the term when the target is there to be found |
| **It does** | a run to look at: ponder chess and read what came out — every rule it proposed, what it weighs, what it was fitted against, and what it cost. If the terms counting queens outweigh the terms counting knights, OMF ordered the pieces itself, and nothing in OMF knows what a queen is |
| **It is used** | **self-play, and look at what it does.** Chess played without a heuristic draws almost every time — the fifty-move rule, a repetition, two bare kings — because nothing is steering toward anything. A few decisive games is the sign that the deduction is doing work; a hundred draws is the sign that it isn't, whatever its fit said |

**Efficiently** is its own measure, and the model registry already keeps what it needs: what a reading of the model
costs in seconds, and what it is worth. A heuristic that reads a hundred terms to say what three would say is
expensive, and the time management policy will drop it for a cheaper one under a clock. The run has to show the cost
beside the deduction, not only the deduction.

None of that is mine to judge. What this step owes is a run whose output can be read and disagreed with: the rules
it deduced, what they cost, and the games it played with them.

**A run with self-play withheld is a test of its own.** Give it the rules, refuse it games, and watch how long it
keeps pondering. What it shows:

- **whether it stops by itself.** A ponderer that runs until its budget is gone, every time, has no idea when it has
  learned what the rules can teach; one that stops early and says why does.
- **where the ceiling of pure deduction is.** Whatever it has when it stops is everything chess's rules alone were
  worth, and the gap between that and what self-play adds is the thing worth knowing.
- **whether the time was well spent.** Seconds against what it came back with, which is the same measure the time
  management policy will apply to pondering as a task.

**This puts self-play on the critical path.** Only games show whether a deduction is used, so the ponderer can't be
shown to work without one, and the self-play the training step needs — the agent loop playing both sides, with a
referee running the game as `openmind-play` does — has to come with it rather than after it.

## Far-fetched heuristics

Searching expressions over a game's readings will turn up far-fetched ones: a conjunction of three conditions over
two grids that fits the target and means nothing. It has to, since the search is free to compose anything.

**Plausibility is the wrong filter**, and it isn't ours to apply. "A rook is worth more on a file with no pawns of
mine" is a real chess heuristic and, written as an expression, is indistinguishable in shape from noise. A filter on
how sensible a rule looks to a person would throw away exactly the findings worth having.

What decides is evidence, and three things already push that way:

- **A term is charged for.** The fit sweeps a price per rule, so a term earns its place or it goes — the cheapest
  explanation that fits, rather than the one that fits best.
- **It is chosen on rows it was not fitted on.** The generator already keeps held-out rows and picks the fit by what
  it does there, which is what separates a rule from a coincidence in the rows it saw.
- **Games have the last word.** A rule that survives both and then changes nothing in self-play is a rule that
  explained the target and not the game.

What is kept is kept as an **opinion with its evidence**, not a belief: where it came from, what it was fitted
against, what it costs to read. A far-fetched rule that keeps paying is a discovery, and the record is what lets it
be re-examined when it stops paying.

## What this step leaves alone

- **Fitting the weights on games**: the training step. This step measures, self-play judges.
- **The bootstrapper that learns which bootstrapping pays off** — which relaxations, which principles, in what order
  — which is a model of its own, trained across games.
- **Move value heuristics**: what a move is worth follows from what the position it leads to is worth, and the search
  reads both; proposing move rules of its own can wait until position rules hold up.

## Where it measures is its own to work out

**Decided (Maxime):** which positions the measurement is made on is not ours to choose. OMF has to work it out, and
where it can't reach a good deduction, the answer is to sharpen its tools rather than to hand it ours.

So the ponderer spends its budget and judges its own measuring as it goes:

- It starts from the position the game starts at, which is the only one it is given.
- It reaches further positions by the game's own actions, since that is the only way any position is reached.
- It watches what it is measuring. A measurement that tells the kinds apart says something; one where everything
  comes out alike — chess's opening position, where every piece is blocked by its own — says nothing yet, and is a
  reason to look further rather than a result.
- It stops when what it measures stops moving, or when the budget runs out, and says in the logs which of the two
  happened and how sure it is.

The weights are the fit's, not anyone's choice: what a reading is worth is what it took to match the targets. If
chess comes out with a term counting queens weighed above a term counting knights, OMF deduced that, and the unit is
whatever the fit landed on.

**Where the deduction comes out poor, that is a finding, not a failure to paper over.** A chess agent that can't tell
a queen from a knight is telling us its readings are too thin or its relaxations too few — the tools to improve, not
a table to hand it.

## Open

Nothing is open; the step waits for your OK.
