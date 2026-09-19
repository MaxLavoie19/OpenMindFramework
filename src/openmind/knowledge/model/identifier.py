import uuid


def new_identifier(kind: str) -> str:
    """A GUID for one entry of that kind: `belief-3f2c9a1e-…`, so an id read alone says what it names."""
    return f"{kind}-{uuid.uuid4()}"
