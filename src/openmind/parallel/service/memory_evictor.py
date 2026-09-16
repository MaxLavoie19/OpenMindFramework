from collections.abc import MutableMapping
from itertools import islice


def evict_oldest(kept: MutableMapping[object, object], keep: int) -> int:
    """Drops the oldest entries of a cache until it holds at most `keep` of them, and gives how many it dropped.

    What a cache remembers is kept in the order it was remembered, so the oldest entries are the ones a search has
    moved on from: dropping those leaves the working set — the positions it is still coming back to — in place. A cache
    emptied wholesale loses that working set and has to derive it again, which is what this is here to avoid."""
    held = len(kept)
    dropping = held - max(keep, 0)
    if dropping <= 0:
        return 0
    if keep <= 0:
        kept.clear()
        return held
    for key in list(islice(iter(kept), dropping)):
        kept.pop(key, None)
    return dropping
