#: A rule's meaning: a tree of plain JSON values, each node a dict whose first key names what it is, such as
#: {"compare": "at least", "left": {...}, "right": {"number": 30}}. See language/README.md for every kind of node.
type Meaning = dict[str, object]
