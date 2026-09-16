# doxastic

## Purpose

What the agent remembers, and where each of its claims stands.

Truth here is not a label. A claim carries two piles of evidence, the records for it and the records against it, kept
apart, so an agent can hold poor evidence for `p` and solid evidence for `not p` — and so can hold both at once and know
that it does. Nothing is overwritten, nothing is averaged away, and everything is kept word for word: what the agent was
told and by whom, what it saw, what it did itself and what came of it, what it proved, what it counted over many games,
and what holds only in a relaxed domain.

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
| `service/knowledge_base.py` | `KnowledgeBase`: `remember`, `cite`, `recall`, `attend`, `weigh`, `claims`, `contested`, `forget` |
| `builder/knowledge_base_builder.py` | `KnowledgeBaseBuilder`: `with_directory`, `with_store` |
| `factory/knowledge_base_factory.py` | `create_knowledge_base(domain, directory=None, record_store=None)` |

## The kinds of source

| Source | What it is | What it is worth |
|---|---|---|
| `proved` | a deduction within the rules | conclusive, where the deduction was complete |
| `seen` | a state or what a player saw of one | as good as the observation |
| `played` | the agent did it and saw what came of it | one case, its own |
| `counted` | many cases folded, such as a signal's agreements | as strong as the count makes it |
| `told` | someone said so, truly or not | as much as the teller is worth |
| `relaxed` | proved in a relaxed domain, where rules were dropped or widened | it holds there; here it is a hint |
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

## Where it applies

Games feed it: `agent/service/game_memory.py` remembers every finished game as it ends, self-play, arms and matches, as
`played` records — each model once, word for word, the game, and each player's outcome — and the training, the
evaluator and the dashboard's models table read from there. The other sources it was built for, each its own step:

- the training's signals and proofs — a signal's agreements and disagreements as `counted` records, a deduction's
  proofs as `proved`, a relaxed domain's as `relaxed`;
- what a player sees, from `StateObserver`, as `seen`;
- what another player said, in games where a claim can be a lie and in rhetoric, as `told`.

## Notes

- Lookup words are matched whatever their case; the record's own words are never touched.
- `EvidenceWeigher` is where discounting belongs when it comes — weighing a teller by how reliable they have been, which
  is where this meets the rhetorical agent's ethos. Nothing is discounted yet.
- The knowledge base logs at INFO what it loads, what it attends to and what it forgets, and at DEBUG each record
  remembered.
- Tests: `model/claim_tests.py`, `model/belief_tests.py`, `mapper/record_json_mapper_tests.py`,
  `service/file_record_store_tests.py`, `service/record_index_tests.py`, `service/recall_cache_tests.py`,
  `service/evidence_weigher_tests.py`, `service/knowledge_base_tests.py`.
