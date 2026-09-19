# Interfaces: epistemology

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-18)

- **Foundherentism** (design, "Epistemology"):
  - Anchors are direct experiences and fundamental rules.
  - OMF strives for coherence between its observations, its models and the rules.
  - A conflict becomes a task.
  - No claim supports itself, and a circle of claims gets no confidence from being a circle.
  - Confidence depends on the method and its measured accuracy, on the amount and representativeness of the data, and on
    how independent the supports are.
  - It is re-evaluated when a method's accuracy is re-measured.
- **Certainty has several models, chosen by the case:**
  - Bayesian for discrete values: the caller's value is the prior, and each piece of evidence updates it through its
    mechanism's error model.
  - Gaussian error for continuous values, where Bayes over named values isn't a proper certainty model.
  - Fuzzy logic over the evidence's strengths where there are no measurements.
- **References:** `Source` gets a field `rests_on: tuple[str, ...]` holding the ids of the experiences, beliefs,
  opinions or rules it rests on.
- **When it runs:** it depends on the context. For some contexts it runs whenever a belief is set; for others it is
  deferred to a task. The time management policy decides.
- **Accuracy:** when a later anchor settles a variable, every mechanism that had produced a belief about it is scored
  right or wrong. A mechanism's accuracy is a belief about that mechanism.

- **Best effort (Maxime):**
  - The agent always does the best it can in the circumstances, using the model that fits best.
  - If no model fits, it lives with it, since there's nothing it can do. It may note that it needs a way to solve that
    kind of problem, and add that as a task.
  - Applied here:
    - A belief no certainty model fits keeps its value and certainty as given, and adds a task: "find a way to assess
      <kind of belief>".
    - Representativeness counts where a model can measure it, and is otherwise left out the same way.
- **Conflicts become warnings and tasks.** The model may have solved a problem incorrectly, and the proper inference may
  be something else.

## Change to `knowledge`

```python
@dataclass(frozen=True, slots=True)
class Source:
    mechanism: str
    parameters: tuple[tuple[str, Value], ...] = ()
    at: datetime | None = None
    rests_on: tuple[str, ...] = ()      # ids of the experiences, beliefs, opinions or rules this rests on
```

Direct experience keeps its raw data as the anchor. `GameMemory` and the continuous trainer's proofs put the game's
experience id in `rests_on`.

## `epistemology`

It depends on `knowledge`.

### Models

```python
@dataclass(frozen=True, slots=True)
class Justification:
    """Where a belief's support comes from, traced back through `rests_on`."""
    belief_id: str
    anchors: tuple[str, ...]            # the anchor ids its support reaches: direct experiences, frozen rules
    circular: tuple[str, ...]           # the ids met again on the way, which give no support
    independent_supports: int           # supports reaching anchors through different paths


@dataclass(frozen=True, slots=True)
class Conflict:
    """Knowledge that doesn't cohere: beliefs on the same variable that disagree, or a belief against an anchor."""
    id: str                             # conflict-<uuid4>
    variable: str
    context: str                        # a context id
    ids: tuple[str, ...]
    kind: str                           # "values disagree", "against an anchor", "against a frozen rule"
```

### Services

```python
class Justifier:
    def justify(self, knowledge: KnowledgeBase, belief: Belief) -> Justification: ...


# Mechanism (id, name, declared_accuracy, tags) lives in knowledge: see rule-world-knowledge.md, "Mechanisms".


class ErrorModel(Protocol):
    def likelihood(self, said: Value, true: Value, accuracy: float) -> float: ...


class CertaintyModel(Protocol):
    """One way to turn a belief's justified evidence into its value, certainty and precision."""
    def fits(self, belief: Belief, error_models: Mapping[str, ErrorModel]) -> bool: ...
    def assess(self, belief: Belief, evidence: tuple[Evidence, ...], error_models: Mapping[str, ErrorModel]) -> Belief: ...

# BayesianCertainty:  discrete values, where the evidence's mechanisms have measured accuracy. The caller's value is the
#                     prior, and each mechanism's error model is the likelihood (spread evenly when it has none).
# GaussianCertainty:  continuous values: each estimate with its mechanism's error as a Gaussian. The posterior's mean is
#                     the value, and its spread is the precision.
# FuzzyCertainty:     no measurements: fuzzy logic over the evidence's strengths.


class CertaintyAssessor:
    """Picks the first certainty model that fits the belief and applies it. Only evidence whose justification reaches
    an anchor counts. When no model fits, the belief stays as given, and a task is added to find a way to assess that
    kind of belief (once per kind)."""
    def __init__(self, models: Sequence[CertaintyModel], error_models: Mapping[str, ErrorModel]) -> None: ...   # error models by mechanism id
    def assess(self, knowledge: KnowledgeBase, belief: Belief) -> Belief: ...     # the belief with its value, certainty and precision set


class AccuracyScorer:
    """When an anchor settles a variable, scores every mechanism that had produced a belief about it."""
    def settle(self, knowledge: KnowledgeBase, variable: str, context: str, anchor_id: str) -> tuple[Belief, ...]: ...   # the mechanisms' updated accuracy beliefs
    def accuracy(self, knowledge: KnowledgeBase, mechanism: str) -> Belief | None: ...


class CoherenceChecker:
    """Every conflict found is logged as a warning and becomes a task: investigate it, since the model may have solved
    the problem incorrectly and the proper inference may be something else."""
    def conflicts(self, knowledge: KnowledgeBase, context: str) -> tuple[Conflict, ...]: ...
    def as_task(self, conflict: Conflict, value: tuple[Belief, ...], expected_time: Belief) -> Task: ...    # "investigate the conflict", to be scheduled


class Epistemology:
    """Makes a context's beliefs rigorous: justify, assess, find conflicts. Called whenever a belief is set, or as a
    task, depending on the context."""
    def review(self, knowledge: KnowledgeBase, belief: Belief) -> Belief: ...
    def audit(self, knowledge: KnowledgeBase, context: str) -> tuple[Conflict, ...]: ...
    def immediate(self, context: str) -> bool: ...     # the per-context setting (E6), until the time management policy decides
```

## Open points

- **E1 (decided): the prior is the caller's value**, at the certainty the caller gave.
- **E1, as asked: the prior.** A Bayesian update needs a prior over the values. What is it before any evidence?
  - Options: uniform over the values the evidence names; the belief's value as set by its caller; a model learned per
    variable or per context.
- **E2 (decided): it depends on the mechanism.** Each mechanism carries its own error model: which wrong values it gives
  when it's wrong.
- **E2, as asked: likelihood with several possible values.** A mechanism's accuracy says how often it's right. When it's wrong,
  which wrong value does it give?
  - Options: spread evenly over the other candidate values; learned per mechanism, as a confusion table.
- **E3 (decided): declared.** Whoever declares a mechanism gives its starting accuracy. Without one, the prior is
  uninformative.
- **E3, as asked: a mechanism never measured.** What accuracy does it start with?
  - Options: an uninformative prior (Beta(1, 1), so 0.5, which moves as soon as it is scored); a value its declarer
    gives.
- **E4 (decided): fuzzy fallback.** In some cases there are no actual measurements, and certainty must fall back on
  fuzzy logic. `strength` is that fuzzy degree, used where the mechanism's accuracy hasn't been measured.
- **E4, as asked: `Evidence.strength`.** With accuracy as the likelihood, what does `strength` still mean?
  - Options: drop it; keep it as the mechanism's own confidence in this one output (such as a decoder's score),
    combined with its accuracy; keep it only for evidence whose mechanism has no measured accuracy.
- **E2, follow-up (decided):** a mechanism without an error model spreads its error evenly over the other candidate
  values.
- **E5 (decided):** in some cases Bayes isn't a proper certainty model, so it isn't used. A Gaussian error may be
  better for continuous values.
- **E5, as asked: numeric and continuous values.** Bayes over "the values the evidence names" works for bool and text. For
  floats, such as a position's value or a time, what is updated?
  - Options: bins, from the binning models (design, "Utility"); a distribution per mechanism, with `precision` as its
    spread.
- **E6 (decided):** a setting per context says immediate or deferred, until the time management policy decides.
- **E6, as asked: before the time management policy exists.** Which contexts review immediately and which defer?
  - Options: a setting per context until the policy is built; always immediate for now.
- **E7 (decided):** whoever finds the conflict sets the task's value and expected time. The next-best-task loop learns
  better ones later.
- **E7, as asked: a conflict's task.** A task needs value and expected-time beliefs. What are they before anything is learned?
  - Options: set by the caller, and left to the next-best-task loop to learn; unset until that loop exists.
