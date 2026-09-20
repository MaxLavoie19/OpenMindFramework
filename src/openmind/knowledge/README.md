# knowledge

## Purpose

Everything the agent knows about a domain, in one knowledge base:

- **Direct experiences**: raw data as received, kept word for word, such as a text, a clip, a reading, an API's
  response or a finished game. Each one has a source: which camera, microphone, API, user or referee it came from.
- **Beliefs**: a variable as some holder takes it to be, holding a value of any type. A belief is held at 100 %
  certainty unless the caller says otherwise, and needs no evidence. Evidence may be given, and it is kept whole: the
  evidence for the value held and the evidence for other values stay apart. A belief also carries an accuracy, which
  depends on the mechanism that produced it, and a precision. The holder is a chain: `()` is the agent's own belief,
  `("black",)` what the agent believes black believes, and `("black", "white")` a level deeper.
- **Opinions**: the agent's own subjective judgements, with any qualifier, and optionally the reasons they rest on.
  Another agent's opinion is known only as a belief held by that agent.
- **Tasks**: something the agent can spend time on, with its value and expected time as beliefs, and a status.
- **Contexts**: compartments of knowledge, named and linked by id, each a level in the hierarchy (`parent`), and the
  contexts a variant took its rules from (`inherits`).
- **Mechanisms**: what evidence comes from, named and linked by id.
- **Rules**: the rules of games, which applications declare, and the heuristics OMF produces. A rule an application
  declares is frozen unless it was declared open. OMF refuses to revise a frozen rule and logs a warning for the
  developer.
- **Models**: the ways a task can be performed in a context, each a record of its own — a ruleset, a lookup table, a
  network — with what it is measured at kept as beliefs about it (see `model/README.md`).
- **Policies, goals and preferences**: how the agent plays, what it wants, and what each goal weighs for whoever holds
  it (see `policy/README.md` and `utility/README.md`).
- **Rulesets**: parts of a context holding the rules for one purpose — the game's simulation, the position value
  heuristic, the move value heuristic, a tactic's, a known agent's play style — each one model of its task. A ruleset
  lists its rules by id, each with its weight there; a rule can be in several rulesets. A frozen ruleset lists only
  frozen rules; changing one means changing an open copy (`copy_ruleset`), and revising a frozen rule in an open ruleset
  revises a copy of it there only (`revise_in`).

Evidence is a source: the mechanism that produced it, its parameters, and the ids of the experiences, beliefs,
opinions or rules it rests on (`rests_on`). A mechanism is whatever produces evidence — a camera, a microphone, an API,
a user, a decoder such as "fork detector", an inference, self-play, an application's declaration — with a name and an
optional declared accuracy. Everything carries tags, key–value pairs it can be retrieved by: topic, date, keywords, and
so on.

Everything has a GUID, `<kind>-<uuid4>` (`belief-3f2c9a1e-…`), and every link uses it. Contexts and mechanisms also have
names, which people and applications use: the knowledge base resolves a name to its id (`context_named`,
`ensure_context`, `mechanism_named`, `ensure_mechanism`), and logs show both (`chess (context-…)`).

Certainty is only as rigorous as its caller makes it. `epistemology` will make it rigorous.

## Content

| File | What it is |
|---|---|
| `model/source.py` | `Source(mechanism, parameters=(), at=None, rests_on=())`: where something came from: the id of the mechanism that produced it, its parameters, and the ids it rests on; `parameter(name)` gives the first value under a name |
| `model/mechanism.py` | `Mechanism(id, name, declared_accuracy=None, tags=())` |
| `model/identifier.py` | `new_identifier(kind)`: a GUID, `<kind>-<uuid4>` |
| `model/evidence.py` | `Evidence(value, strength, source)`: a source's support for one value |
| `model/belief.py` | `Belief(variable, context, value, holder=(), certainty=1.0, accuracy=None, precision=None, evidence=(), tags=(), id="")`; `key` (variable, context, holder), `supporting` and `opposing` |
| `model/direct_experience.py` | `DirectExperience(variable, context, value, source, at=None, tags=(), id="")`: raw data, word for word |
| `model/opinion.py` | `Opinion(variable, context, value, at=None, reasons=(), tags=(), id="")`: the agent's own judgement and the sources it rests on |
| `model/task.py` | `Task(name, context, value, expected_time, status="pending", tags=(), id="")` |
| `model/context.py` | `Context(id, name, parent=None, inherits=(), tags=())`, `parent` and `inherits` holding ids |
| `model/rule_record.py` | `RuleRecord(name, kind, rule, source, action=None, parameter=None, probability=1.0, open=False, tags=(), id="")`: a rule belongs to a context only through the rulesets listing it |
| `model/ruleset.py` | `Ruleset(name, context, task, source, links=(), open=False, tags=(), id="")`: a part of a context holding the rules for one purpose, one model of its task; `rule_ids`, `weight(rule_id)` |
| `model/policy.py` | `Policy(name, context, sub_goal="", position_value="", move_value="", tags=(), id="")`: a way of playing tuned to a subset of the actions, naming the models performing its two tasks; the default policy has no name |
| `model/goal.py` | `Goal(name, context, tags=(), id="")`: something an agent wants in a context |
| `model/preference.py` | `Preference(goal, weight, holder=(), role="", tags=(), id="")`: what a goal weighs for whoever holds it, in a role |
| `mapper/policy_json_mapper.py` | `PolicyJsonMapper`: policies, goals and preferences to JSON and back |
| `model/model_record.py` | `ModelRecord(name, task, context, family, mechanism, location="", tags=(), id="")`: one way to perform a task — a ruleset, a lookup table, a network — sourced by the mechanism its readings are scored through, and found again by `location` |
| `mapper/model_record_json_mapper.py` | `ModelRecordJsonMapper`: a model record to JSON and back |
| `constant/task_constant.py` | The tasks OMF knows, each with a port of its own: simulation, prediction, position value, move value, tactic value, inference, binning, abstraction, agent model, planning, time management |
| `model/ruleset_link.py` | `RulesetLink(rule, weight=1.0)`: a rule a ruleset lists, with its weight there (linear for now) |
| `mapper/ruleset_json_mapper.py` | `RulesetJsonMapper`: a ruleset to JSON and back |
| `model/tags.py` | `Tags`, key–value pairs; `carries(tags, wanted)` |
| `model/store.py` | `Store`: the port one kind of knowledge is kept behind, as JSON-ready dicts: `append`, `load`, `forget` |
| `constant/knowledge_constant.py` | The knowledge directory and its files, the kinds ids start with, the names of the mechanisms OMF registers itself (`direct experience`, `declaration`, `inference`, `decoder`, `self-play`), and task statuses |
| `constant/rule_kind_constant.py` | The kinds of rules: a game's (`GAME_KINDS`), heuristics (`HEURISTIC_KINDS`) and `abstraction`, what one level of the hierarchy sees of a state |
| `mapper/knowledge_json_mapper.py` | `KnowledgeJsonMapper`: experiences, beliefs, opinions, tasks, contexts and sources to JSON-ready dicts and back |
| `mapper/rule_record_json_mapper.py` | `RuleRecordJsonMapper`: a rule to JSON and back; a rule given as a function is found again by its module and name |
| `service/file_store.py` | `FileStore(path)`: the default `Store`, JSON lines appended to one file; the last line of an id wins |
| `service/knowledge_base.py` | `KnowledgeBase`: `experience`, `experienced`, `experiences`; `believe`, `belief`, `belief_by_id`, `beliefs`; `hold`, `opinion`, `opinions`; `task`, `tasks`; `context`, `ensure_context`, `context_named`, `context_by_id`, `contexts`, `readable_context`; `mechanism`, `ensure_mechanism`, `mechanism_named`, `mechanism_by_id`, `mechanisms`, `readable_mechanism`; `model`, `model_by_id`, `model_named`, `models`, `readable_model`; `policy`, `policy_by_id`, `policy_named`, `policies`, `readable_policy`; `goal`, `goal_by_id`, `goal_named`, `goals`, `readable_goal`; `prefer`, `preference`, `preferences`; `declare`, `revise`, `frozen`, `rule`, `undeclare`; `ruleset`, `ruleset_by_id`, `ruleset_named`, `rulesets`, `ruleset_rules`, `link`, `unlink`, `frozen_ruleset`, `copy_ruleset`, `revise_in`, `readable_ruleset`. Contexts are passed as ids |
| `builder/knowledge_base_builder.py` | `KnowledgeBaseBuilder`: `with_directory`, `with_store(kind, store)` |
| `factory/knowledge_base_factory.py` | `create_knowledge_base(domain, directory=None, stores=None)` |

## Usage

```python
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.direct_experience import DirectExperience
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.source import Source

knowledge = create_knowledge_base("cheat")
cheat = knowledge.ensure_context("cheat").id
microphone = knowledge.ensure_mechanism("table microphone").id
reader = knowledge.ensure_mechanism("claim reader").id
heard = knowledge.experience(DirectExperience("heard", cheat, "three kings", Source(microphone)))
knowledge.believe(
    Belief(
        "black is bluffing",
        cheat,
        True,
        certainty=0.7,
        evidence=(Evidence(True, 0.7, Source(reader, rests_on=(heard.id,))),),
        tags=(("topic", "bluffs"),),
    )
)
knowledge.beliefs(tags=(("topic", "bluffs"),))
```

A domain's knowledge is kept under `data/knowledge/<domain>/`, one JSON-lines file per kind: `experiences.jsonl`,
`beliefs.jsonl`, `opinions.jsonl`, `tasks.jsonl`, `contexts.jsonl`, `mechanisms.jsonl`, `rules.jsonl`, `rulesets.jsonl`, `models.jsonl`, `policies.jsonl`, `goals.jsonl` and `preferences.jsonl`. An integrator gives their own
storage with `with_store(kind, store)` and owns the data's lifecycle.

## Logs

- `openmind.knowledge.service.knowledge_base`:
  - `INFO Knowledge of <domain>: <n> direct experiences, <n> beliefs, <n> opinions, <n> tasks, <n> contexts, <n> mechanisms, <n> rules, <n> rulesets, <n> models, <n> policies`
    when a base opens with knowledge already kept.
  - `WARNING Rule <name> (<id>) is frozen: …` when a revision of a frozen rule is refused; `WARNING Ruleset <name> (<id>)
    is frozen …` when a frozen ruleset would list an open rule, or is asked to revise one.
  - `INFO Copied ruleset …`, and `INFO Rule … is frozen: revised as its copy … in ruleset … only`.
  - `DEBUG` a line for every experience, belief, opinion, task, context and rule kept.
- `openmind.knowledge.service.file_store`: `INFO Forgot <id> in <file>`.
