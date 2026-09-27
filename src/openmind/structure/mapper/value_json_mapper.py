from openmind.structure.model.record import Record

#: How a record is written down: what kind of record it is, and what its parts hold.
RECORD, PARTS = "record", "parts"


class ValueJsonMapper:
    """Maps a value a game hands OMF to JSON-ready data and back.

    **What a game hands OMF is either one of OMF's own formats or it says how to write itself.** A record is one
    of them, so a game whose squares are records is writing in a format OMF knows. Anything that is neither is
    refused here rather than at the moment of writing, where what went wrong is a line of JSON and no longer a
    value anybody can name.

    **It lives beside the record rather than beside any one thing that writes one.** The same values reach disk
    by more than one road — a rule saying what a parameter may take, a consequence saying what a square becomes
    — and each road that answered this for itself answered it differently or not at all. A promotion drawn as
    `Always(Piece('white', 'queen'))` was written straight out as a Python object and brought a run down two
    hours after it started, while the road beside it had known how to write a record for days."""

    def to_data(self, value: object) -> object:
        """That value as JSON takes it: a name, a number, nothing, or a record written as its parts."""
        if isinstance(value, Record):
            return {RECORD: value.kind, PARTS: {name: self.to_data(one) for name, one in value.parts}}
        if isinstance(value, tuple | list):
            return [self.to_data(one) for one in value]
        if value is None or isinstance(value, str | int | float | bool):
            return value
        raise TypeError(
            f"{value!r} is not one of OMF's values: a name, a number, nothing, a record, or a list of those. "
            "A value a game hands OMF either uses one of its formats or says how to write itself down."
        )

    def from_data(self, data: object) -> object:
        """That value again, a record made as the game declared it."""
        if isinstance(data, dict) and RECORD in data:
            parts = data.get(PARTS) or {}
            return Record.of(str(data[RECORD]), {name: self.from_data(one) for name, one in parts.items()})
        if isinstance(data, list):
            return [self.from_data(one) for one in data]
        return data
