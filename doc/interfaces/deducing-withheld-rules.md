# Interfaces: deducing rules an integrator didn't declare

Waiting for Maxime's OK. Nothing is coded until then.

A game is declared by someone who may know their game far better than they know OMF, and the shortest way to make a
game work is to hand over a list of legal actions. That works — OMF plays — and it quietly takes away everything OMF
reasons with: no constraint to relax, no rule to read, no way to ask what a piece could do if something were
different. Chess declared that way cannot be bootstrapped, as this session showed.

So OMF has to notice, say so, and deduce what it can.

## Noticing

Two signs, both readable from what was declared:

| Sign | What it means |
|---|---|
| an action whose legality no constraint decides — its values rules already give only legal actions | the rules are generated, not declared |
| an action whose parameters are single opaque values — one string standing for the whole action | the action has no parts to reason about |

The first is checked by solving: where the candidates the values rules offer and the actions the constraints allow
are the same set, across a few positions, nothing is being constrained.

## Saying so

A warning, once per game, naming what is lost rather than scolding:

> `chess` generates its legal actions rather than declaring what makes them legal. OMF will play, but it cannot
> relax the rules, reason about what a piece reaches, or bootstrap a heuristic from them. It will try to deduce them.

And for the second sign:

> `move` has one parameter, `uci`, whose values are whole actions. OMF cannot read a part of it, so nothing it infers
> can be about how a move works. Declaring the parts — where it starts, where it lands — is what lets it reason.

## Deducing, where the parts are there

Given an action whose parameters are readable, OMF can ask the game questions it already knows how to ask: make a
position, list the legal actions, and see which candidates were allowed. Every candidate that was offered and not
allowed is an example of the rule it is missing.

That makes it a classification, not a fit over payoffs: **what tells the allowed candidates from the rest?** The
expression search already composes candidates out of readings and keeps what explains the data at a price; what is
new is what it reads.

```python
class RuleDeducer:
    """What tells a game's legal actions from its illegal ones, as rules it can then relax and reason with."""

    def deduce(self, knowledge_base, game, action: str, settings) -> tuple[RuleRecord, ...]: ...
```

**The missing vocabulary.** Every reading today is about a position. Deducing a constraint needs readings about the
candidate as well:

- what each of its parameters is, and what the position holds there — the piece on the source, the piece on the
  target, whose they are
- how its parameters stand to one another — the offset between two cells, the distance, whether they share a row, a
  column or a diagonal
- what lies between them

Those are grid readings applied to an action's parameters rather than to a cell, so the shapes exist; what doesn't is
the generator reading an action at all.

**Where the positions come from.** The same walk the ponderer uses, and positions built on purpose — a board with one
piece on it says more about how that piece moves than a hundred positions from play. Building positions is the
capability the reach measurement wanted too.

## What it gives back

Rules declared in the game's context, in a ruleset of their own, sourced as deduced rather than declared. They are
**opinions**: they explain every candidate OMF has seen and may still be wrong. What they are for is relaxation and
reasoning — the game keeps playing by the integrator's own enumerator, which is faster and is not in doubt.

## What can't be recovered

An action whose parameters are opaque. OMF can learn which whole actions are allowed in which positions and nothing
more: `"e2e4"` has no parts, and inventing a meaning for its letters would be OMF knowing chess.

## Open

1. **How sure it must be to declare a rule.** A rule that explains every candidate seen may fail on the next
   position. Either it is declared as an opinion at once and revised when a candidate contradicts it, or it is held
   back until it has survived some number of positions. The first fits the doxastic side, which is built to hold
   opinions and revise them; the second declares less nonsense.
2. **Whether it deduces constraints only, or effects too.** What a move *does* is as withheld as what makes it legal,
   and it is deducible the same way — from positions before and after. It is a larger job, and everything bootstrapped
   so far needs only legality.
