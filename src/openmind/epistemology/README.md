# epistemology

## Purpose

Makes the agent's beliefs rigorous, by foundherentism. A belief's support is traced back through its sources'
`rests_on` to its anchors: direct experiences, and rules an application declared (frozen rules). A circle of beliefs
resting on each other gives no support. Certainty is then assessed from the evidence that reaches an anchor, by the
first certainty model that fits:

- **Bayesian**, for a value that isn't a number, where every mechanism's accuracy is known (measured, or declared). The
  caller's value is the prior, at the caller's certainty. Each piece of evidence updates it through its mechanism's
  error model, spread evenly over the other values where the mechanism has none of its own. The candidates are every
  value the evidence names, plus "anything else".
- **Gaussian**, for a number, where every mechanism's spread on numbers has been measured. The estimates are combined,
  each weighed by the inverse of its variance. The value is their mean, and the precision their combined spread.
- **Fuzzy**, where there are no measurements. A value is supported as much as its strongest evidence, and held as much
  as it is supported and not opposed.

Best effort: a belief with no justified evidence, or one no model fits, stays as given. The agent lives with it, and a
task to find a way to assess that kind of belief is added once. A mechanism's accuracy is measured when an anchor
settles a variable: every piece of evidence about it scores its mechanism. Conflicts become warnings and tasks, since a
model may have solved a problem incorrectly.

## Content

| File | What it is |
|---|---|
| `model/justification.py` | `Justification(belief_id, anchors, circular, independent_supports, justified)`: the anchors a belief's support reaches, the ids met again on the way, and which pieces of evidence reach an anchor |
| `model/conflict.py` | `Conflict(id, variable, context, ids, kind)` |
| `model/error_model.py` | `ErrorModel`: `likelihood(said, true, accuracy, candidates)` |
| `model/certainty_model.py` | `CertaintyModel`: `name`, `fits(knowledge, belief, evidence)`, `assess(knowledge, belief, evidence)` |
| `constant/epistemology_constant.py` | The kinds of conflict, the "anything else" candidate, and the names of the beliefs kept about mechanisms |
| `service/justifier.py` | `Justifier.justify(knowledge, belief)` |
| `service/even_error_model.py` | `EvenErrorModel`: right as often as its accuracy, evenly wrong otherwise |
| `service/accuracy_scorer.py` | `AccuracyScorer`: `settle(knowledge, variable, context, anchor_id)` scores the mechanisms; `accuracy(knowledge, mechanism_id, context)` measured, else declared, else None; `spread(knowledge, mechanism_id, context)` |
| `service/bayesian_certainty.py` | `BayesianCertainty(accuracy_scorer, error_models=None)` |
| `service/gaussian_certainty.py` | `GaussianCertainty(accuracy_scorer)` |
| `service/fuzzy_certainty.py` | `FuzzyCertainty()` |
| `service/certainty_assessor.py` | `CertaintyAssessor(models)`: `assess(knowledge, belief, justification)`, `fitted`, `unassessed` |
| `service/coherence_checker.py` | `CoherenceChecker(justifier)`: `conflicts(knowledge, context)`, `as_task(knowledge, conflict, value, expected_time)` |
| `service/epistemology.py` | `Epistemology`: `review(knowledge, belief, need_value=(), need_time=None)`, `audit(knowledge, context, value, expected_time)`, `immediate(context)` |
| `factory/epistemology_factory.py` | `create_epistemology(error_models=None, immediate=None)` |

## Usage

```python
from openmind.epistemology.factory.epistemology_factory import create_epistemology

epistemology = create_epistemology()
kept = epistemology.review(knowledge, belief)                       # assessed, then believed
conflicts = epistemology.audit(knowledge, context_id, value, expected_time)
```

## Logs

- `openmind.epistemology.service.certainty_assessor`: `INFO <variable> in <context>: <value> at <certainty> by the
  <model> model, from <n> of <m> pieces of evidence`.
- `openmind.epistemology.service.accuracy_scorer`: `INFO <mechanism> said … where … turned out true: right|wrong,
  accuracy …` and `INFO <mechanism> was off by …: its spread is …`.
- `openmind.epistemology.service.coherence_checker`: `WARNING Conflict in <context> on <variable>: <kind>, between …`.
- `openmind.epistemology.service.epistemology`: `WARNING No certainty model fits <kind> in <context>: kept as given`.
- `openmind.epistemology.service.justifier`: `DEBUG <variable> rests on <n> anchors …`.
