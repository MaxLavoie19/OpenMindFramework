from openmind.structure.model.record import Record

#: What a data structure holds: something scalar, nothing, or a record of named parts.
#:
#: Everything here is immutable and hashable, which is what lets a state be compared, used as a search key and sent
#: to another process. A record is the one way a cell holds more than one thing about one thing, and it keeps those
#: promises because it is frozen.
type Value = str | int | float | bool | None | Record
