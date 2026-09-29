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
    def standing(self, ruleset, decisions, rater) -> Mapping[str, Standing]: ...


@dataclass(frozen=True, slots=True)
class Standing:
    """What a rule is worth as a member of a ruleset, and how much is known about it.

    `went_with_winning` and `moved_the_decision` are what it is worth; `fired` is how much is known, and is
    never folded into either. A rule that fires three times and one that fires three hundred can be worth the
    same, and only the second is known to be."""

    went_with_winning: float
    moved_the_decision: float
    fired: int
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

## Two numbers, and they are not rivals

An earlier draft of this treated the fitted weight and a rule's contribution as two answers to one question.
They are not, and the distinction is the whole of how a mate detector works.

**What a rule weighs in a ruleset** is `RulesetLink.weight`, the multiplier in the value sum — how loudly the
rule speaks when it speaks. A checkmate finder needs an enormous one, so that on the rare position where it
fires it swamps every other term and the move is taken. It is a fact about the ruleset's arithmetic.

**What a rule is worth as a member** is what decides whether it stays and what its vouching signal earns. It
is read from two things:

- **how strongly it went with winning** — the probability it put on the move that was played, weighted by what
  the game paid whoever played it;
- **how much it moved the decision** — how much worse the set does with it left out.

**How often it fires is neither.** It is how much is known about the rule, not how good it is. Three firings
and three hundred can show the same worth, and the second is held far more confidently — so the count governs
how fast a rule's standing moves and how long before it may be dropped, and never its standing itself. That
is the settled rule kept rather than bent: *"coverage is reported beside the score and never folded into it. A
rule that fires in three per cent of positions and is right is a rule worth keeping; folding coverage into the
score destroys exactly those."* A mate detector fires in well under one position in a hundred, and marking it
down for that is precisely the error.

**The multiplier was the open question, and it is now measured.** An L1 price shrinks large coefficients
hardest, and a term that fires rarely must speak loudly when it does — so it needs exactly the coefficient the
price penalises most, while paying off on the fewest rows. Measured, holding the decisive strength at 0.5 and
varying only how often the term fires:

```
fires in    price 0   price 0.001   price 0.01
      1%      0.501         0.382        0.000     <- dropped
      5%      0.503         0.481        0.278
     20%      0.499         0.493        0.436
     50%      0.499         0.495        0.459
```

**At a mate detector's rate the term is dropped outright at 0.01, which is in the run's own default sweep** —
while weaker but commoner terms survive. Rarity alone decides it. This is "a detector marked down for being a
detector" happening in the fitting, where the settled rule only protects it in the scoring.

**The cause is what the price is charged on.** `ExpressionSearch.share` is *"the share of rows where the
column isn't blank"* — blank meaning nothing was read, not nothing was found. A mate detector reads **0** on
almost every position, which counts as speaking, so it is charged as though it spoke everywhere.

**Charging it on how often it fires fixes it, with the `costs` the fitter already takes:**

```
price   priced as now   priced on firing
0.001           0.382              0.500
0.01            0.000              0.489
0.05            0.000              0.442
```

This is the adaptive lasso — Zou, JASA 101:1418, whose answer to the lasso's selection inconsistency is to
weight each coefficient's penalty rather than charge them all alike. It is not an invention here.

- **Open, and smaller than it was:** the present cost is `clauses × share × what it takes to read`, and the
  read cost is a real and separate thing — a look-ahead term reads the position after every legal action and
  genuinely costs that on every position, fired or not. Replacing `share` with the firing rate must not
  quietly forgive the reading. **Undecided** whether the firing rate replaces `share` or multiplies beside it.

## The vouching mechanism

`RuleBudget` and `RulePrice` are built. What is not is the thing that spends: who bids, where the gate sits,
and what pays the signals back. This section is the one waiting for an OK.

### Where the gate sits

`HeuristicFinder._declared` links **every** non-zero term of the chosen fit into the ruleset. The gate goes
exactly there, between "the fit kept it" and "it is linked". Everything a signal could want to read is already
in hand at that line: the term's readings on the training rows, the payoffs those rows led to, the weight the
fit gave it, and the terms admitted before it.

> **This is a second gate, after L1, and that is deliberate.** The price sweep decides what a term *weighs*;
> the budget decides whether it is *admitted*. A term the fit zeroed never reaches a signal, so the budget can
> only ever let through fewer rules than the fit kept — which is the direction open question 41 is about.

### The port

```python
class RuleSignal(Protocol):
    """A way of proposing that a rule earns its place.

    A signal rates rather than bids. What it wants is on whatever scale suits it; how much it gets is its
    budget, which is the point of a budget over a weight — a signal cannot talk its way to more by rating
    louder."""

    @property
    def name(self) -> str: ...

    def rates(self, candidates: Sequence[Candidate]) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    """One term the fit kept, and everything a signal may read about it. Every field is already computed by
    the sweep that produced it; nothing here costs a second pass over the rows."""

    expression: Expression
    rule: PythonRule
    weight: float          # what the fit gave it
    readings: np.ndarray   # its column on the training rows
    payoffs: np.ndarray    # what those rows led to


class RuleAdmission:
    """Which candidates got vouched for, and by whom.

    Each signal walks its own candidates in the order it rates them and buys while it can afford to, so a
    signal spends on what it wants most first. A candidate several signals bought is admitted once and
    credited to all of them, which is the share map `RuleBudget.earned` already takes."""

    def admitted(
        self, knowledge_base, context_id: str, candidates: Sequence[Candidate],
        signals: Sequence[RuleSignal], vocabulary: int,
    ) -> Mapping[int, tuple[str, ...]]: ...
```

### The roster

All of them, because they are different signals and some will suit some games — "the signal that gives the
best heuristic for a problem is weighted more than others", and here that weighting is what it earns.

| Signal | What it wants | Strand |
|---|---|---|
| `went with winning` | terms whose readings line up with the payoff on the rows they fired on | beats-chance |
| `moved the fit` | terms whose absence costs the most held-out loss — leave-one-out, arithmetic over readings already taken | marginal contribution |
| `says something new` | terms least explained by the terms already admitted | novelty |
| `fires often enough to know` | terms that fired on enough rows for their record to mean something | confidence |

**The fourth one needs saying out loud, because it looks like the error this project has a standing rule
against.** "Coverage is reported beside the score and never folded into it" — a mate detector marked down for
being rare is precisely the mistake. A signal that prefers common terms does not mark anything down: it
declines to spend, and three other signals are still free to buy the mate detector. If preferring common terms
is a bad way to pick rules, this signal earns less and buys less, which is the economy working. **If you would
rather it were not there at all, say so and it goes** — the other three stand without it.

### What pays them

After a heuristic has played and been judged, `earned` is called once with each signal's share of that
ruleset and the ruleset's worth. That is `_judged` in the chess repo, which already computes `mass` against
`offered` per position. Nothing new is measured.

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
