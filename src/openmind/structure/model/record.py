from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:  # A value may be a record, so the alias imports this; typing it here would close the circle.
    from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Record:
    """A value with named parts, such as a piece that has a colour and a type.

    A cell holds one value, and some games have nothing scalar to put in one: a chess square holds a white knight,
    which is two things about one thing rather than one thing. Without this a game must either fuse them into a
    name like `"white knight"` — after which nothing can ask what colour it is without knowing every piece there
    is — or spread one thing across two grids that must then be kept in step.

    **Frozen and slotted, and its parts are values.** A grid holding these must stay hashable, comparable and
    picklable, since states are compared, used as search keys and sent to other processes. An object that could
    change would break every one of those quietly, so what may go in a cell is this and not anything at all.

    A game declares its own by subclassing:

        @dataclass(frozen=True, slots=True)
        class Piece(Record):
            colour: str
            type: str

    OMF never looks inside one except to read its parts by name, and never assumes what those names are.

    **A record can be written down, because a record is one of OMF's own formats.** A value a game hands OMF
    is either one OMF already knows — a name, a number, nothing, or one of these — or it comes with its own way
    of being written down and read back. Anything else cannot be kept: a game whose squares were a class of its
    own had every rule about them silently lost on the way to disk, and the loss showed up as a game that
    declared perfectly and could not be opened again.

    Each subclass is known by the name it is declared under, so what is read back is the game's own record and
    not a shape standing in for one. Two games declaring a record of the same name are the same game's, which
    is true of everything else a context holds."""

    #: Every kind of record a game has declared, by name, so one written down can be made again.
    _kinds: ClassVar[dict[str, type["Record"]]] = {}

    #: The names this already uses, which a game's part may therefore not take.
    _TAKEN = ("kind", "parts", "_kinds")

    def __init_subclass__(cls, **named: object) -> None:
        # Refused at the moment the game declares it, because the alternative is silence. A part named `kind`
        # shadows the property saying what kind of record this is, so the record writes itself down under
        # whatever that part happens to hold — a piece whose type is a queen is written as a record called
        # `queen` — and comes back, much later and somewhere else, as a KeyError about a record no game ever
        # declared. Nothing between the two says anything at all.
        clashing = [one for one in cls._TAKEN if one in cls.__dict__.get("__annotations__", {})]
        if clashing:
            raise TypeError(
                f"{cls.__name__} declares a part called {clashing[0]!r}, which is a name a record already "
                f"uses for itself; a game's parts may not be called any of {list(cls._TAKEN)}."
            )
        # Registered without calling up, because a frozen slotted dataclass is built twice — once as written
        # and once by the decorator — and the second is the class a game actually holds. The later
        # registration is the one that stands, which is the one wanted.
        Record._kinds[cls.__name__] = cls

    @property
    def parts(self) -> tuple[tuple[str, "Value"], ...]:
        """Each part's name and what it holds, in the order the game declared them."""
        return tuple((one.name, getattr(self, one.name)) for one in fields(self))

    @property
    def kind(self) -> str:
        """What the game calls this kind of record."""
        return type(self).__name__

    @staticmethod
    def kinds() -> dict[str, type["Record"]]:
        """Every kind of record a game has declared, by name.

        What a rule reading a written-down position needs: it names the classes the position holds, and for a
        game of parts those are the game's own."""
        return dict(Record._kinds)

    @staticmethod
    def of(kind: str, parts: "Mapping[str, Value]") -> "Record":
        """A record of that kind again, from its parts; raises KeyError where no game declared one.

        A game must have been declared for its records to be read back, which is true of its rules as well —
        a knowledge base is opened for a game, and the game says what its values are."""
        made = Record._kinds.get(kind)
        if made is None:
            raise KeyError(f"No game has declared a record called {kind!r}; there are {sorted(Record._kinds)}")
        return made(**parts)  # type: ignore[arg-type]
