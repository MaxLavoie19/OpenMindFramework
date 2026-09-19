#: Where a domain's knowledge is kept, under `<domain>/`.
KNOWLEDGE_DIRECTORY = "data/knowledge"

#: The files a domain's knowledge is kept in, one JSON object per line, in the order it was written.
EXPERIENCES_FILE = "experiences.jsonl"
BELIEFS_FILE = "beliefs.jsonl"
OPINIONS_FILE = "opinions.jsonl"
TASKS_FILE = "tasks.jsonl"
CONTEXTS_FILE = "contexts.jsonl"
MECHANISMS_FILE = "mechanisms.jsonl"
RULES_FILE = "rules.jsonl"

#: What each kind of entry's id starts with, before its UUID, so an id read alone says what it names.
EXPERIENCE = "experience"
BELIEF = "belief"
OPINION = "opinion"
TASK = "task"
RULE = "rule"
CONTEXT = "context"
MECHANISM = "mechanism"
CONFLICT = "conflict"

#: The names of the mechanisms OMF itself registers. Any other mechanism is registered by whoever uses it.
DIRECT_EXPERIENCE = "direct experience"
DECLARATION = "declaration"
INFERENCE = "inference"
DECODER = "decoder"
SELF_PLAY = "self-play"

#: A task's status.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
TASK_STATUSES = (PENDING, RUNNING, DONE)
