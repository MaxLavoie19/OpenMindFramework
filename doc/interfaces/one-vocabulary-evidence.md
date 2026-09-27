# What the literature says about a universal world description language

Evidence for [one-vocabulary.md](one-vocabulary.md). Gathered by a fan-out of search, fetch and adversarial
verification: 56 extracted claims were each put to verifiers instructed to break them, against primary sources
downloaded and extracted locally rather than read in summary. **34 were refuted and 22 survived**, all of the
survivors at high confidence. What follows is only what survived.

## Somebody has already learned rules in the event calculus

**ILED — Katzouris, Artikis and Paliouras, *Incremental Learning of Event Definitions with Inductive Logic
Programming*, Machine Learning 100(2–3):555–585, 2015.** Event definitions learned incrementally by inductive
logic programming, in the event calculus. This is the closest published thing to what this project is
building, and its concessions are the interesting part:

- **The dialect is fixed, not learned.** *"We call the Event Calculus dialect used in this work Simplified
  Discrete Event Calculus (SDEC)."* A restricted, discrete dialect — equivalent to the classical one but
  simplified — chosen so learning is tractable. Nobody learned in the full calculus.
- **The correctness guarantee is bought with assumptions.** Supervision is assumed **noise-free and complete**.
- **The scale it was shown at**: nine interrelated high-level events over 1000 time points, with seven at one
  level of nesting, one at a second and one at a third.

For this project that is both encouraging and a warning. The representation is learnable — but the published
result assumes perfect, complete supervision, which is exactly what a learner watching a game does *not* have,
and it works in a deliberately cut-down dialect rather than the whole thing.

## The predicates themselves can be learned, not only the rules

**SIFT — Gösgens, Jansen and Geffner, *Learning Lifted STRIPS Models from Action Traces Alone: A Simple,
General, and Scalable Solution*.** From traces alone, with no predicates supplied: *"it involves learning the
domain predicates as well."*

This bears directly on the open question about what a reading is. It is evidence that a fixed hand-written
vocabulary is not the only option — the alternative has been built and scaled.

## Learning an action model can use what is refused as well as what is allowed

**AMLSI and TempAMLSI — Grand, Pellier and Fiorino.** *"AMLSI takes as inputs two training datasets, I+ and
I−, and outputs a PDDL domain. I+ (positive samples) contains the observed feasible state/action sequences,
and I− (negative samples) contains infeasible state/action sequences computed from I+."* Learned by grammar
induction; TempAMLSI extends it to temporal action models, and notes that in non-sequential domains some
non-temporal structure has to be handled specially.

Two learners here, one from what the game allows and one from what it refuses, is the same shape — and this
says the pairing is standard rather than eccentric.

**N-SAM and N-SAM\*** learn into a *hand-designed* target — numeric PDDL 2.1, Boolean fluents plus numeric
functions — and the learned model is then consumed unchanged by off-the-shelf planners. Its representational
commitment is explicit: numeric preconditions are the **convex hull** of the observed pre-states, emitted as
the linear inequalities defining that hull, and numeric effects come from linear regression over pre and post
values. Convex and linear, by choice, not by accident.

## The cautionary tale has numbers

**Cyc — Lenat and Marcus, *Getting from Generative AI to Trustworthy AI: What LLMs might learn from Cyc*,
arXiv 2308.04445, 31 July 2023.**

- *"it has taken a coherent team of logicians and programmers four decades, 2000 person-years, to produce the
  current Cyc KB"*, and **"Cycorp's experiments with larger-sized teams generally showed a net decrease in
  total productivity"** — it does not parallelise.
- Footnote 9, written by Lenat himself, and the end of it is the part that matters: *"We noticed empirically
  that the general theorem-proving reasoner actually took so long that over a million queries in a row that
  called on it, as a last resort, just timed out. Going back farther, we saw that that had happened for
  decades. **So, about one decade ago, we quietly turned the general theorem prover off, so it never gets
  called on! The only impact is that Cyc sometimes runs a bit faster, since it no longer has that attractive
  but useless nuisance available to it.**"* Verified 3–0.
- **And the architecture that replaced it is this plan's architecture.** Verified 3–0: Cyc separates *"the
  epistemological problem — what does the system know?"* from *"the heuristic problem — how can it reason
  efficiently?"*. One clean expressive language, CycL, for what is known; *"multiple redundant, specialized
  reasoners — Heuristic Level (HL) modules — each of which is much faster than general theorem-proving when
  it applies. By 1989, Cyc had 20 such high-level reasoners; today it has over 1,100."*
- **Nothing was ever learned.** Verified 2–0: tens of millions of assertions hand-authored, *"each axiom is
  hand-checked for default correctness, generality, and best placement"*.

**What was refuted is as important as what stood.** A claim that Lenat named one representation for all
knowledge as the project's central design error was refuted **0–3**: the same paper defends the commitment —
*"only higher order logic can represent the same breadth of thought as a natural language"* — and frames
Cyc's speed as a triumph. So the honest reading is narrower and much more useful to this plan:

**The universal vocabulary survived as the surface. What died was the single reasoner over it, and what
replaced it was eleven hundred specialised representations.** That is not a verdict against one vocabulary.
It is a verdict for exactly the shape this plan proposes — one language for what is known, many cheap
specialised representations for what is computed — and a warning about the *number*. Cyc reached 1,100 HL
modules and added one whenever an application proved too slow. This plan's flat searchable readings are HL
modules by another name, and the thing to watch is whether they stay derived from the vocabulary or start
multiplying beside it.

## The neuro-symbolic claim is weaker than it is stated

**OpenCog Hyperon — Goertzel et al., arXiv 2310.18318.** Its Atomspace is *"the primary meta-representational
hub"* — but there is *"a broader Space API that permits the creation of multiple specialized types of Spaces
within it"*, and the account of neural integration is, in the paper's own words, that *"you could wrap a large
language model or other deep neural networks in an Atomspace API, and perform pattern matching against"* them.

**Wrapping, not unifying**, and the paper says so. The most ambitious current claim to one universal
representation is in fact a hub with heterogeneous spaces behind an interface.

## Neither major cognitive architecture can handle time

**Laird, *An Analysis and Comparison of ACT-R and Soar*, Advances in Cognitive Systems 2021 (arXiv
2201.09305), sole author and Soar's creator.** Both have at most an optional module estimating *short*
durations; **neither can judge longer time scales**, and Laird names this an open research problem.

Two architectures, four decades of work between them, and the thing this project proposes to put at the centre
of its vocabulary — a moment, and one time before another — is the thing they name as unsolved.

## What this changes in the plan

1. **The event calculus choice is supported, and its price is now known.** It has been learned in, by
   inductive logic programming, but in a *restricted dialect* and under noise-free complete supervision.
   Expect to restrict the dialect deliberately rather than discover the restriction later.
2. **Learning the predicates is a live option**, not a fantasy — SIFT does it from traces alone.
3. **Cyc's failure mode is the one to watch, and it is not the one usually quoted.** The representation
   survived; reasoning over it generally did not. A universal vocabulary that nothing can afford to reason
   over is the specific risk, which is what the flat searchable readings in the plan exist to avoid.
4. **Nobody has demonstrated one vocabulary shared across perception, prediction and planning.** The strongest
   claimant wraps. That makes this ambitious rather than late.
5. **Time is an open problem in the field, not a solved thing to adopt.** The plan should say so.

## Caveats about this evidence

- Two thirds of what was extracted did not survive — the refutation rate was high, and mostly for overreach:
  a quote real but load-bearing beyond what it says, a scope widened, a permission read as universality.
- The run was interrupted by permission prompts and its own synthesis step did not complete. This is written
  from the 86 completed agents' outputs directly.
- Coverage is uneven: the Cyc, event-calculus-learning and action-model-learning angles are well evidenced;
  the neural world-model line (Dreamer, JEPA, Genie) and the arguments-against angle are thin here, and what
  survived on them is not enough to conclude from.
