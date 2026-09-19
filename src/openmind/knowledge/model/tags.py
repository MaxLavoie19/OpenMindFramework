from openmind.structure.model.value import Value

#: What an entry of the knowledge base can be retrieved by: key–value pairs such as ("topic", "openings"),
#: ("date", "2026-09-18") or ("keyword", "fork"). A key may appear more than once.
type Tags = tuple[tuple[str, Value], ...]


def carries(tags: Tags, wanted: Tags) -> bool:
    """Whether the tags carry every wanted pair."""
    return all(pair in tags for pair in wanted)
