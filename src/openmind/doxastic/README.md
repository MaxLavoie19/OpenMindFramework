# doxastic

## Purpose

What the agent remembers, where each of its claims stands, and every rule it knows.

Truth here is not a label. A claim carries two piles of evidence, the records for it and the records against it, kept
apart, so an agent can hold poor evidence for `p` and solid evidence for `not p` — and so can hold both at once and know
that it does. Nothing is overwritten, nothing is averaged away, and everything is kept word for word: what the agent was
told and by whom, what it saw, what it did itself and what came of it, what it proved, what it counted over many games,
and what holds only in a relaxation.

Doxastic, not epistemic: a belief can be false, and a claim can be about another player's belief — `holder` is a chain,
so `("black",)` is what the agent believes black believes and `("black", "white")` one level deeper.

## Content

| File | What it is |
|---|---|
| `model/claim.py` | `Claim(name, rule=None, about=(), holder=())`: something that can be true or false; `rule` reads it on a position, `about` is who or what it concerns, `holder` whose belief it is; `key` identifies it |
| `model/provenance.py` | `Provenance(source, told, at, game, round, ply, when)`: where a record came from |
| `model/record.py` | `Record(text, provenance, subjects, names, keywords, claim, supports, strength, count, id)`: one thing remembered, word for word |
| `model/belief.py` | `Belief(claim, supporting, opposing)`: `belief`, `disbelief`, `uncertainty`, `contested`, `net` and `by_source`; `accumulated` is the arithmetic |
| `model/record_store.py` | `RecordStore`: the port a store answers — `append`, `read`, `load`, `forget` |
| `constant/doxastic_constant.py` | the kinds of source, the knowledge directory and the records file |
| `mapper/record_json_mapper.py` | `RecordJsonMapper`: a record as one line of JSON, and back |
| `service/file_record_store.py` | `FileRecordStore`: records as JSON lines in one append-only file, a record's place its byte offset |
| `service/record_index.py` | `RecordIndex`: which records to find by subject, name, keyword, claim, teller and source; holds ids, not records |
| `service/recall_cache.py` | `RecallCache`: the records in context, emptied when the process holds more than its share of memory |
| `service/evidence_weigher.py` | `EvidenceWeigher`: which records bear on a claim, and which way |
| `model/rule_record.py` | `RuleRecord(name, kind, rule, provenance, contexts=(), action=None, parameter=None, probability=1.0, id="")`: a rule the agent knows, with its weight in each context it bears on and, for an effects rule, the chance its outcome happens; `weight(context)` and `relevant(context)` |
| `constant/rule_kind_constant.py` | The kinds of rule, what a game project declares (`GAME_KINDS`) and what the inference engine produces (`HEURISTIC_KINDS`), and the rules file |
| `model/rule_store.py` | `RuleStore`: the port a rule store answers — `append`, `read`, `load`, `forget` |
| `mapper/rule_record_json_mapper.py` | `RuleRecordJsonMapper`: a rule as one line of JSON, and back; source keeps its source, a function keeps where it lives and is imported again |
| `service/file_rule_store.py` | `FileRuleStore`: rules as JSON lines in one append-only file, a rule's place its byte offset |
| `service/knowledge_base.py` | `KnowledgeBase`: `remember`, `cite`, `recall`, `attend`, `weigh`, `claims`, `contested`, `forget`; `declare`, `rules(context, kinds=())`, `rule`, `contexts()`, `undeclare` |
| `builder/knowledge_base_builder.py` | `KnowledgeBaseBuilder`: `with_directory`, `with_store`, `with_rule_store` |
| `factory/knowledge_base_factory.py` | `create_knowledge_base(domain, directory=None, record_store=None, rule_store=None)` |

## The kinds of source

| Source | What it is | What it is worth |
|---|---|---|
| `proved` | a deduction within the rules | conclusive, where the deduction was complete |
| `seen` | a state as it was | as good as the observation |
| `played` | the agent did it and saw what came of it | one case, its own |
| `counted` | many cases folded, such as a signal's agreements | as strong as the count makes it |
| `told` | someone said so, truly or not | as much as the teller is worth |
| `relaxed` | proved in a relaxation, a game with fewer constraints | it holds there; here it is a hint |
| `assumed` | taken to be so, for want of anything better | until something better comes |

## How a record is kept

Records go into a store and never move. `FileRecordStore` writes one JSON line per record, only ever appending: writing
a record again under its id makes the newest line win, which is how a counted record raises its count, and forgetting
one writes a line saying so. A record's place is its byte offset, so citing it later is a seek and a line, however long
the file has grown. `data/knowledge/<domain>/records.jsonl`, unless a domain is given another directory.

`RecordStore` is a port: SQLite, or anything else, takes its place with `KnowledgeBaseBuilder.with_store` and the
knowledge base never knows the difference.

The index holds ids and places, so it stays small as the store grows; the cache holds the records in context. Attending
to a subject, a name or a keyword brings everything under it into the cache, where recalling it costs nothing; like the
reading cache, it empties when the process holds more than its share of memory, and a record let go of is still in the
store.

## The rules the agent knows

Every rule lives here, whatever wrote it: a game project declares the rules of its game, and the inference engine
declares the heuristics it generates. Whatever plays then retrieves the rules relevant to its context, and an RBS built
for it (see `rbs/README.md`) is the game — `base.rules("chess")` gives the rules of chess, its heuristics included,
and `base.rules("chess", (CONSTRAINT,))` only what makes a move legal.

| Kind | What the rule says |
|---|---|
| `initial` | where the game starts |
| `players` | who plays, the variable naming the player to act, the payoff variables |
| `empty` | what a grid's cell holds when nothing is on it |
| `definitions` | the names a context's rules all see |
| `constraint` | whether an action with these parameter values is legal |
| `values` | the values a parameter of an action can take |
| `effects` | what an action leads to, with the chance it happens |
| `ending` | why a finished game ended |
| `record` | a game's record, from its initial state and the actions played |
| `timeout` | what a player's clock running out does to the payoffs |
| `picture` | a position as an image |
| `position` | a position's value for a player: a position heuristic's term |
| `move` | a move's value for the player to act: a move heuristic's term |

A rule carries a **weight per context**, not one weight: the same rule can be heavy in one game and light in another,
and a context it has no weight in is one it says nothing about. Declaring a rule again under its id changes what it
weighs, the way a counted record raises its count, and the newest line wins. A variant, a relaxation, a round or an arm
is a context whose rules are its game's plus its own: its game's rules gain a weight there rather than being copied.

Rules are kept in `data/knowledge/<domain>/rules.jsonl`, beside the records. A rule written as source keeps its source;
a rule the project gave as a function keeps the module it lives in and its name there, and loading imports it again, the
same way a worker process finds it. A function that can't be found raises `ValueError` rather than coming back missing:
a game short of a rule is a different game.

## Usage

```python
from openmind.doxastic.constant.doxastic_constant import SEEN, TOLD
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record

base = create_knowledge_base("cheat")
no_spade = Claim("black holds no spade", about=("black",))

base.remember(Record('black said "I have no spades"', Provenance(TOLD, "black"), ("black",),
                     keywords=("claim",), claim=no_spade, supports=True, strength=0.3))
base.remember(Record("black played the spade queen", Provenance(SEEN, at="trick-4"), ("black",),
                     claim=no_spade, supports=False, strength=1.0))

belief = base.weigh(no_spade)
belief.belief, belief.disbelief, belief.contested   # 0.3, 1.0, 0.3
belief.by_source                                     # (('seen', 0.0, 1.0), ('told', 0.3, 0.0))

base.attend("black")                                 # everything about black, in context
base.recall(told="black")                            # what black has said, word for word
base.contested()                                     # the claims held both ways at once
```

Rules:

```python
from openmind.agent.factory.tictactoe_factory import declare_tictactoe
from openmind.doxastic.constant.rule_kind_constant import CONSTRAINT

declare_tictactoe(base)                  # the game project declares its rules

base.rules("tictactoe")                  # every rule of the game, the heaviest first
base.rules("tictactoe", (CONSTRAINT,))   # only what makes a move legal
```

## Where it applies

Games feed it: `agent/service/game_memory.py` remembers every finished game as it ends, self-play, arms and matches, as
`played` records — each model once, word for word, the game, and each player's outcome — and the training, the
evaluator and the dashboard's models table read from there. The other sources it was built for, each its own step:

- the training's signals and proofs — a signal's agreements and disagreements as `counted` records, a deduction's
  proofs as `proved`, a relaxation's as `relaxed`;
- a position as it was, as `seen`;
- what another player said, in games where a claim can be a lie and in rhetoric, as `told`.

## Notes

- Lookup words are matched whatever their case; the record's own words are never touched.
- `EvidenceWeigher` is where discounting belongs when it comes — weighing a teller by how reliable they have been, which
  is where this meets the rhetorical agent's ethos. Nothing is discounted yet.
- The knowledge base logs at INFO what it loads, what it attends to and what it forgets, and at DEBUG each record
  remembered and each rule declared.
- Tests: `model/claim_tests.py`, `model/belief_tests.py`, `mapper/record_json_mapper_tests.py`,
  `service/file_record_store_tests.py`, `service/record_index_tests.py`, `service/recall_cache_tests.py`,
  `service/evidence_weigher_tests.py`, `service/knowledge_base_tests.py`,
  `mapper/rule_record_json_mapper_tests.py`, `service/file_rule_store_tests.py`.
