# Interfaces: rule, world, knowledge

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-18)

- **Beliefs:** a belief can hold a value of any type: bool, text, int, float, … It is a variable with a certainty and
  its sources.
- **Evidence:** evidence is a source. A source is a mechanism with its parameters: inference, a decoder, direct
  experience, and so on. The evidence for a value and the evidence for other values are kept apart.
- **Contexts:** there are two links. A context has a parent, the level it sits in, which is compartmentalized. It may
  also inherit from other contexts, such as a variant sharing its game's rules.
- **The old code is discarded:** its source kinds (proved, seen, played, counted, told, relaxed, assumed) and its
  `Provenance` came from an agent that didn't understand the design.
- **Rules:** `Rule` and `PythonRule` move to a `rule` package, which whoever needs them imports.

## `rule`

It depends only on `world`.

| Module | Content |
|---|---|
| `rule/model/python_rule.py` | `PythonRule(source: str)`: moved unchanged from `rbs` |
| `rule/model/rule.py` | `type Rule = PythonRule \| Callable[..., object]`, and the protocols `ConstraintRule`, `ValuesRule`, `EffectsRule`, `EndingRule` and `RecordRule`: moved unchanged from `rbs` |

Every import of `openmind.rbs.model.rule` and `openmind.rbs.model.python_rule` is repointed. Nothing else changes.

## `world`

This step leaves it unchanged: `State`, `Action`, `JointAction`, `Players`, `Value`, `Grid`. The grid moves to
`structure` in that component's step.

**Observations (decided W1, W2).** An observation is something that comes from a source: an API, a decoder, or any
other input. It isn't a construct beyond a belief: it is a direct experience, kept in `knowledge` (see
`DirectExperience`).

## `knowledge`

It depends on `world` and `rule`. It no longer imports `rbs`.

### Models

```python
@dataclass(frozen=True, slots=True)
class Source:
    """A piece of evidence: the mechanism that produced it and its parameters.
    Direct experience:  Source("direct experience", (("experience", "<DirectExperience id>"),), at)
    A decoder:          Source("decoder", (("decoder", "fork detector"), ("state", "<id>")), at)
    An inference:       Source("inference", (("method", "deduction"), ("plies", 3), ("premises", ("<belief id>", ...))), at)"""
    mechanism: str
    parameters: tuple[tuple[str, Value], ...]
    at: datetime


@dataclass(frozen=True, slots=True)
class DirectExperience:
    """Raw data as received, kept word for word: a text, a clip, a reading, a response from an API, … Nothing rewrites
    it. It has the same shape as a belief, but needs no support and no certainty. Data extracted from it are beliefs
    whose sources reference it. It replaces today's `Record`."""
    variable: str
    context: str
    value: Value                     # the raw data, or a reference to where it is stored
    source: Source                   # which camera, microphone, API, user, …
    at: datetime
    tags: tuple[tuple[str, Value], ...] = ()
    id: str = ""


@dataclass(frozen=True, slots=True)
class Evidence:
    """A source's support for one value of a variable, with how much it carries (0 to 1)."""
    value: Value
    strength: float
    source: Source


@dataclass(frozen=True, slots=True)
class Belief:
    """A variable as some holder takes it to be, in one context.
    holder: () is the agent's own; ("black",) is what the agent believes black believes; ("black", "white") goes a level
    deeper. The evidence is kept whole: the evidence for `value` and the evidence for other values stay apart."""
    variable: str
    context: str
    holder: tuple[str, ...]
    value: Value
    certainty: float = 1.0           # how strongly it is held; 100 % unless specified (see K1)
    accuracy: float | None = None    # kept within the belief; it depends on the mechanism that produced it
    precision: float | None = None   # how narrow the value is, such as the spread of an estimate; None where it doesn't apply
    evidence: tuple[Evidence, ...] = ()   # optional; epistemology makes it rigorous
    tags: tuple[tuple[str, Value], ...] = ()   # for retrieval: ("topic", "openings"), ("date", "2026-09-18"), ("keyword", "fork"), …
    id: str = ""


@dataclass(frozen=True, slots=True)
class Opinion:
    """Like a direct experience: subjective, with any qualifier as its value ("good", "cheap", …). It may have reasons:
    "I don't like winter" because of the opinion "I don't like the cold" and the belief "winter is cold". The agent
    edits only its own opinions. Another agent's opinion is known only as a Belief whose holder is that agent."""
    variable: str                    # what it is about, such as "winter"
    context: str
    value: Value                     # the qualifier, such as "don't like"
    at: datetime
    reasons: tuple[Source, ...] = ()   # optional: the beliefs, opinions or experiences it rests on
    tags: tuple[tuple[str, Value], ...] = ()
    id: str = ""


@dataclass(frozen=True, slots=True)
class Context:
    """A compartment of knowledge, and a level in the hierarchy.
    parent:   the level it sits in; its knowledge isn't shared.
    inherits: contexts whose rules it takes on purpose, such as a variant from its game."""
    name: str
    parent: str | None = None
    inherits: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Task:
    """Something OMF can spend time on, in a context.
    value: Beliefs of its worth over several measures, weighed together by preferences: utility gained, precision gained,
    time saved. They drift as runs of the task report what they brought. expected_time is a Belief too."""
    name: str
    context: str
    value: tuple[Belief, ...]
    expected_time: Belief
    status: str                      # pending, running or done
    id: str = ""


@dataclass(frozen=True, slots=True)
class RuleRecord:
    """A rule the knowledge base holds: as today, with `source: Source` in place of `provenance`, and one new field.
    open: False by default for rules an application declares, which stay frozen. A rule or ruleset declared open, or
    a rule OMF deduced, may be revised."""
    name: str
    kind: str
    rule: Rule
    source: Source
    contexts: tuple[tuple[str, float], ...] = ()
    action: str | None = None
    parameter: str | None = None
    probability: float = 1.0
    open: bool = False
    id: str = ""
```

### Service: `KnowledgeBase`

```python
class KnowledgeBase:
    # contexts
    def context(self, context: Context) -> Context: ...                      # registers or updates a context
    def contexts(self) -> tuple[Context, ...]: ...
    # rules (as today, with source and open)
    def declare(self, rule: RuleRecord) -> RuleRecord: ...
    def rules(self, context: str, kinds: tuple[str, ...] = ()) -> tuple[RuleRecord, ...]: ...   # those weighing in it
    def rule(self, rule_id: str) -> RuleRecord | None: ...
    def revise(self, rule_id: str, rule: RuleRecord) -> RuleRecord: ...      # refused for a frozen rule: the refusal is logged as a warning
    # beliefs
    def believe(self, belief: Belief) -> Belief: ...                         # sets or updates a belief, as given
    def belief(self, variable: str, context: str, holder: tuple[str, ...] = ()) -> Belief | None: ...
    def beliefs(
        self, context: str | None = None, holder: tuple[str, ...] | None = None, tags: tuple[tuple[str, Value], ...] = ()
    ) -> tuple[Belief, ...]: ...                                             # every belief carrying all the given tags
    # direct experience
    def experience(self, experience: DirectExperience) -> DirectExperience: ...   # kept word for word, given an id
    def experienced(self, experience_id: str) -> DirectExperience | None: ...
    # opinions
    def hold(self, opinion: Opinion) -> Opinion: ...                         # only the agent's own
    def opinions(self, holder: str, context: str) -> tuple[Opinion, ...]: ...
    # tasks
    def task(self, task: Task) -> Task: ...                                  # adds or updates a task
    def tasks(self, context: str | None = None, status: str | None = None) -> tuple[Task, ...]: ...
```

Storage stays behind store ports, like today's `RecordStore` and `RuleStore`. JSON lines are the default, and an
integrator replaces them with their own storage.

## Open points

- **K11 (decided):** a direct experience has a source: which camera, microphone, API, user, and so on.
- **K12 (decided): variants copy, with a link.** A variant copies its game's rules, less those it leaves, as
  `RuleDeclarer.inherits` does today. `Context.inherits` records the link. `rules(context)` gives the rules weighing in
  that context.

- **K10 (decided, clarified by Maxime):** direct experiences, opinions and beliefs all belong to the knowledge base. A
  direct experience has the same shape as a belief, but needs no support or confidence. An opinion is like a direct
  experience, but may have reasons.

- **K9 (decided): finished games.** Each finished game is stored as a direct experience: its moves and result are the
  raw data. Its payoffs, its players' models and its ending are beliefs referencing it. `GameMemory`, the continuous
  trainer, the game study and the dashboard's game browser and model scores move to that.

- **K8 (decided):** beliefs, direct experiences, opinions, tasks and rules carry tags so they can be retrieved easily:
  topic, date, keywords, and so on. Tags are key–value pairs, and a lookup matches whatever carries all the given
  tags.

- **K1 and K2 (decided):** without epistemology, a belief has 100 % certainty unless specified. It needs no source,
  though sources may be provided. The caller sets the value and the certainty. Epistemology later makes this rigorous.
- **K3 (decided):** accuracy is kept within the belief, and depends on the mechanism that produced it.
- **K4 (decided):** raw data is kept as a direct experience; a clip is a direct experience. Data extracted from it
  reference its source, certainty and so on. `Record` is replaced by `DirectExperience`.
- **K4, as asked:** today's `Record`, one thing remembered word for word. With raw input carried in a direct-experience Source,
  is `Record` still needed, or is it replaced by Source + Evidence?
- **K5 (decided):** there is no fuzzy type in this step. Converting between words and values comes with training: it
  is a trained model, like the heuristics, and may be a rule of the rule base that uses them.
- **K5, as asked:** the fuzzy value type (words as fuzzy sets over a scale, decided in B5). Should `Value` gain a fuzzy form in
  this step, so beliefs can hold one, or when `utility` is built?
- **K6 (decided):** pending, running and done are enough for now.
- **K7 (decided):** the existing data in `data/knowledge/` is dropped. It isn't properly populated.

## Amendment (2026-09-18): GUIDs

Decided by Maxime:
- Everything the knowledge base keeps gets a GUID with a kind prefix: experiences, beliefs, opinions, tasks, rules,
  contexts, and in `epistemology` mechanisms and conflicts.
- A context gets a GUID, and its name stays a readable field.
- Every link uses the GUID. Human-readable logs show the name too.

Proposed:
- **Id form (decided: long names, no abbreviations):** `<kind>-<uuid4>`, with the kinds `experience`, `belief`,
  `opinion`, `task`, `rule`, `context`, `mechanism` and `conflict`. For example,
  `belief-3f2c9a1e-5b7d-4c1a-9e8f-0a1b2c3d4e5f`. They replace today's counters (`b000001`, …).
- **`Context(id, name, parent=None, inherits=(), tags=())`:** `parent` and `inherits` hold context ids.
- **Links hold context ids:** `Belief.context`, `DirectExperience.context`, `Opinion.context`, `Task.context`, a rule's
  per-context weights (`RuleRecord.contexts`), `Source.rests_on`, and the RBS's `context`.
- **Names are resolved at the edges.** An application or a command names a game (`tictactoe/fourinarow`), and the
  knowledge base gives its context:
  - `context_named(name) -> Context | None`;
  - `ensure_context(name, parent=None) -> Context`, which registers the context the first time it is named.

  `RuleDeclarer(knowledge, "tictactoe")` resolves the name this way.
- **Logs** write a context as `name (id)`, and other entries as their id plus a readable part, such as a belief's
  variable.

### Mechanisms (2026-09-18)

Decided by Maxime: every mechanism has a name and a GUID, and each decoder has a name. A decoder is a mechanism.

Proposed:

```python
@dataclass(frozen=True, slots=True)
class Mechanism:
    """Whatever produces evidence: a camera, a microphone, an API, a user, a decoder ("fork detector"), an inference
    ("deduction"), self-play, an application's declaration. `declared_accuracy` is the starting accuracy whoever
    declares it gives; None leaves it uninformative."""
    id: str                             # mechanism-<uuid4>
    name: str
    declared_accuracy: float | None = None
    tags: Tags = ()
```

- `Source.mechanism` holds the mechanism's id. `Source(mechanism_id, parameters, at, rests_on)`.
- Mechanisms are a kind the knowledge base stores, in `mechanisms.jsonl`:
  - `mechanism(mechanism) -> Mechanism` registers or updates one;
  - `mechanism_named(name) -> Mechanism | None`;
  - `ensure_mechanism(name, declared_accuracy=None) -> Mechanism` registers it the first time it is named.
- OMF's own mechanisms are registered by name the first time they're used: direct experience, declaration, inference,
  self-play.
- Error models are code, not data, so they stay in `epistemology`, which is given them by mechanism id.
- A frozen rule is one whose source's mechanism is the declaration mechanism, and that isn't open.
