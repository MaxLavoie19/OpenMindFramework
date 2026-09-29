# Interfaces: what a rule costs and who pays for it

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked
**Open** isn't decided.

## What this is for

A fit keeps every term whose weight survives the sweep — a hundred and fifteen of them in one measured
ponder — and every one is registered, judged against every game that ended, and crossed to every worker. That
is the pool that open questions 41 and 42 are about.

The question underneath is which terms earn their place. Three answers were considered and two were rejected
for reasons worth keeping:

- **A per-rule chance test.** "Admit a rule that would have steered us well and beats chance." This is
  Samuel's 1959 scheme, which he reported unstable: 20 of 38 terms took the top coefficient across 28 games,
  two with reversed signs. Jensen and Cohen measured why — searching 30 candidates on pure noise inflates the
  best apparent score 5.6-fold, and their three aggravating conditions (near-independent candidates, small
  samples, near-equal scores) are exactly a sparsely-firing rule.
- **Weights over ways of making.** Several admission signals, each drawn in proportion to how good its
  heuristics have been. Rejected by Maxime as a majority tyranny: the leading signal wins nearly every draw
  and the rest stop contributing at all, so a signal that is right about a narrow class of rules never gets
  to say so.

This is the third: **a budget**. Signals earn, rules cost, and a signal spends what it has earned to vouch
for a rule. A poor signal earns less and so buys rarely, but it saves and it is never shut out.

It is also what the literature points at. A description-length budget over the ruleset is the only mechanism
in the surveyed corpus that is an explicit utility trade rather than a test; it carried RIPPER's error ratio
from 2.08 to 0.98, and KRIMP used it to take 945,000 candidates down to 282.

## The economy

**A signal earns by the heuristics it vouched for.** The currency is the one everything else here is judged
in: what a heuristic expected of what happened, against what knowing nothing would have expected — `mass`
against `offered`, which the judging computes every position already.

**A signal that does worse than ignorance loses budget.** Income is the measured worth and it is signed, so a
persistently wrong signal is driven out of the economy rather than idling in it. It can be bankrupted by a bad
run, and that is the intended sharpness rather than an accident of it.

**A rule costs what it costs to write down.** Longer rules are dearer, so they exist and are rarer — which is
the whole of what "more clauses are more expensive" should mean.

> **The price is a code, not a formula, and this project has paid for the difference.**
> `doc/learner-practices.md` records the first attempt — `per_clause + len(body) ** 2` — and why it was wrong:
> *"Under a real code a body of k conditions from a pool of C costs `L_N(k) + log₂C(C,k)`, whose marginal cost
> per further condition **falls** with k; under `1 + k²` it **rises**. So the quadratic over-prefers short
> constraints."* `ConstraintSelector` already prices in bits under `--pricing code`; the same code serves here,
> and nothing invents a penalty.

**Buying is vouching, not owning.** A bought rule is one a signal has staked its budget on. It enters the
ruleset and then stands on its own record: it participates in decisions, and it can be dropped for performing
poorly or for being superseded. What a signal buys is the rule's *admission*, not its tenure.

## The interfaces

```python
class RuleBudget:
    """What each signal has earned, and what it has spent vouching for rules.

    A signal is a way of proposing that a rule earns its place. Its budget is what the heuristics it vouched
    for turned out to be worth, and it spends that budget to vouch for more. A signal that does worse than
    knowing nothing loses budget and eventually cannot buy; one that is right about a narrow class of rules
    buys rarely and is never shut out, which is what a weight could not give."""

    def earned(self, knowledge_base, context_id: str, signal: str, worth: float) -> float: ...
    def held(self, knowledge_base, context_id: str, signal: str) -> float: ...
    def afford(self, knowledge_base, context_id: str, signal: str, price: float) -> bool: ...


class RulePrice:
    """What a rule costs to vouch for: the bits it takes to write down, over the vocabulary it is written in.

    Not a penalty chosen here. A body of k conditions drawn from a pool of C costs what naming it costs, so
    the marginal cost of a further condition falls with k — the opposite of what a quadratic does, and the
    reason the quadratic was struck."""

    def priced(self, rule, vocabulary, allowance: float = 1.0) -> float: ...


class RuleTenure:
    """Whether a rule that was bought still earns its place, once it has been making decisions.

    **Vouching gets a rule in; its contribution keeps it there and sets what it weighs.** A rule fires where
    it fires, and it is read over the decisions it actually had an opinion about — never against how often it
    fired. A rule with a better contribution weighs more, and that holds however rarely it fires: when mate is
    on the board it is worth taking, and mate being rare is no argument for taking it less seriously.

    **Superseded means the ruleset does better without the rule than with it.** Not that another rule takes
    its content in, which is a fact about the rules rather than about the playing — a rule can be logically
    redundant and still carry its weight, and a rule nothing subsumes can still be dead weight. So it is
    measured by leaving it out, on the decisions the whole set was judged over, in the same currency.

    **A rule that changes nothing when removed goes, because it is not free to keep.** Reading it costs
    processing on every position of every decision, so a set without it is as accurate and faster — which is
    better on the ordering this project already uses, where `ModelRegistry.best` takes the highest accuracy
    and gives ties to the fastest. Redundancy is sometimes robustness and sometimes nothing, and which it is
    on any particular rule is not foreseeable; what settles it is whether the set as a whole is better off,
    speed included."""

    def spared(self, ruleset, decisions, rater) -> tuple[str, ...]: ...
    def contributed(self, ruleset, decisions, rater) -> Mapping[str, float]: ...
```

## What this replaces

`FactorWeights` goes. Its bound and its softmax draw were for choosing *one* way of making a heuristic, and
there is no drawing when every signal bids. What carries over is the part worth keeping: credit flowing from a
measured heuristic back to what produced it, and the ledger living in the knowledge base so a restart resumes
what the last run learned.

## Leaving one out is nearly free, which is what makes this affordable

Measuring "better without it" reads as one pass per rule — a hundred and sixteen passes for a set of a
hundred and fifteen, at the thirty-three seconds one judging took, which would be an hour. It is not.

**A heuristic's value is the sum of its rules' weighted readings.** So a ruleset's value without one rule is
its value *minus that rule's weighted reading*, and every leave-one-out score falls out of the same readings
the whole set already needed. Read each term once per position, and the set's score and all of its
leave-one-out scores are arithmetic over those numbers.

The cost is therefore about one full judging, not one per rule — and the thing that dominates it, drawing the
position each move leads to, is paid once whatever is being measured.

## Where a rule's weight comes from, which is now two answers

**This is the one thing the decisions above leave in conflict, and it is not mine to settle.**

Today a term's weight is its fitted coefficient: `HeuristicFinder` sweeps an L1 price over position rows and
keeps the fit that predicts held-out *positions* best. The weight is whatever minimises that loss.

"A rule with a better contribution weighs more" is a different quantity, measured on different evidence — how
much the ruleset's score over *decisions* falls when the rule is left out. A term can be excellent at
predicting what a position was worth and contribute little to choosing a move, and the reverse.

**The mate example is exactly where they part.** A term reading "mate is available" is nearly constant across
positions, so a fit over position values has almost no variance to pay it for and shrinks it. Its
contribution when it fires is the whole game. Fitted weight says small; contribution says decisive.

This project has met the same shape before and measured it. `PonderSettings.walks` is off by default, and the
note says why: ordering terms by how steadily they read made held-out loss *worse* every seed — 0.0105
measuring nothing, 0.0357 ordering by steadiness, 0.1252 keeping only the steadiest — because "a term that
never varies along a walk can vary plenty over the positions being fitted". Reading a term's worth off the
wrong sample is a mistake this codebase has already made once.

- **Options, none chosen:**
  - **Contribution replaces the fitted weight.** The fit proposes terms, the decisions weigh them. Truest to
    "rules that have a better contribution weight more", and it throws away a number chosen on held-out rows.
  - **Contribution scales the fitted weight.** The fit sets the shape and the decisions correct it. Keeps
    both measurements and needs a rule for how they combine, which is a formula nobody has evidence for.
  - **Two heuristics, and the games decide.** The same terms weighted each way are two models of one task,
    and the registry already ranks models by what they did. No formula, one more candidate per ponder.
- **Undecided.**

## The tuning knob

**One budget on the command line sets how much comes through.** Income is scaled by an allowance, so a larger
allowance buys more rules and a smaller one fewer, without any threshold being chosen inside. That is the
"tune the budget to allow for more rules" lever, and it is a budget in the project's sense — the caller's to
set, not a constant buried in a service.

## Open

1. ~~What a bad vouch costs.~~ **Decided (Maxime): a bad vouch decays the budget, and it never reaches
   nought.** Decay rather than a charge, so a signal that has been wrong buys less often without being
   bankrupted by a run of bad luck — and a floor above nought, so no signal is ever permanently shut out.
   That is the same reason the weights were rejected: a signal must always be able to come back.
2. ~~What "superseded" means.~~ **Decided (Maxime): the ruleset without the rule performs better than the
   ruleset with it.** Measured by play, not by logic — so `Subsumer` and `RuleReasoner.entails` are the wrong
   instruments, and so is SIRUS's linear-combination test. Both answer whether a rule is *redundant*, which
   is a different claim: a redundant rule is kept here, because the evidence about it says only that it does
   no harm.
3. ~~How a rule's own record is read when it fires rarely.~~ **Decided (Maxime): on the times it fired.**
   No estimate correction and no coverage term. A rule so rare that sets without it outperform sets with it
   vanishes by the same leave-one-out that judges every other rule, and nothing else is needed.
4. ~~Whether the record needs adjusting for how many rules were looked at.~~ **Decided (Maxime): the
   adjustment is where the search is, not where the record is.** The factors assess every rule that has not
   been vouched for, and a vouch is paid for — so what bounds the search is the budget rather than a
   significance level. Once a rule is in a ruleset it is judged by its contribution to that set, which is not
   a test over a candidate space and needs no correction for one.
5. **None of the evidence behind any of this comes from self-play.** Every number above is from i.i.d.
   benchmarks. Positions within a game are correlated and share one terminal payoff, so the effective sample
   size behind "this rule fired five times" is unknown here. That is exactly the independence Minsky made his
   condition for credit assignment, and measuring it would be new evidence rather than borrowed.
