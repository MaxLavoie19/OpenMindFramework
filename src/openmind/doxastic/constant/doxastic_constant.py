#: A deduction proved it, within the rules of the domain.
PROVED = "proved"

#: It was observed: a state, or what a player saw of one.
SEEN = "seen"

#: The agent did it itself and saw what came of it.
PLAYED = "played"

#: Counted over many cases, such as a signal's agreements and disagreements.
COUNTED = "counted"

#: Someone said so, whether or not they were telling the truth.
TOLD = "told"

#: A deduction proved it in a relaxed domain, where some rules were dropped or widened, so it holds there and maybe here.
RELAXED = "relaxed"

#: Taken to be so without evidence, to be held until something better comes along.
ASSUMED = "assumed"

#: Where a record can come from.
SOURCES = (PROVED, SEEN, PLAYED, COUNTED, TOLD, RELAXED, ASSUMED)

#: Where a domain's records are kept.
KNOWLEDGE_DIRECTORY = "data/knowledge"

#: The file holding the records of a domain, one JSON object per line, in the order they were remembered.
RECORDS_FILE = "records.jsonl"

#: How a claim's name, holder and subjects are joined into the key identifying it.
KEY_SEPARATOR = "|"

#: How a holder chain and a claim's subjects are joined inside that key.
CHAIN_SEPARATOR = ">"
SUBJECT_SEPARATOR = ","

#: Records remembered between two readings of this process's memory, above which the cache is emptied.
MEMORY_CHECK_INTERVAL = 1000
