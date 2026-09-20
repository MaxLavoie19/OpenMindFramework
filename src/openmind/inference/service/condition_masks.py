import logging
from collections.abc import Mapping, Sequence

from openmind.inference.service.rule_deducer import Condition
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)


class ConditionMasks:
    """Which actions a condition holds of, kept as one number with a bit for each.

    A rule is grown by asking every condition it might take how much it rules out, and answering that by reading
    every action again — so the same question is asked of the same action once per condition per step, and with
    four hundred thousand actions and a few thousand conditions it is asked more times than there is time for.

    Nothing about an action changes while a rule is grown, so the answer is worked out once: a condition becomes
    the set of actions it holds of, and a set of actions is a number with one bit raised for each. Counting how
    many of them are still in play is then a bitwise and and a count of bits, which the machine does whole words at
    a time, and narrowing a rule is one more and.

    What each reading holds is worked out for every action up front, since that is a single pass and every
    condition about a value is an or of those. A condition comparing two readings to each other is worked out the
    first time it is asked for and kept, because only a handful of the pairs are ever worth asking about and
    working out all of them costs more than the whole search."""

    def __init__(self, readings: Sequence[Mapping[str, Value]]) -> None:
        self._readings = readings
        self.count = len(readings)
        self._held: dict[str, dict[Value, int]] = {}
        self._compared: dict[Condition, int] = {}

    @property
    def everything(self) -> int:
        """The set of every action there is."""
        return (1 << self.count) - 1

    def holding(self, condition: Condition) -> int:
        """The actions that condition holds of."""
        reading, relation, value = condition
        if relation in ("== reading", "!= reading"):
            return self._against(reading, str(value), relation == "== reading")
        held = self._values(reading)
        if relation == "==":
            return held.get(value, 0)
        if not isinstance(value, int | float) or isinstance(value, bool):
            return 0
        wanted = 0
        for one, mask in held.items():
            if isinstance(one, int | float) and not isinstance(one, bool):
                if one <= value if relation == "<=" else one >= value:
                    wanted |= mask
        return wanted

    def _values(self, reading: str) -> dict[Value, int]:
        """Each value that reading takes, with the actions it takes it of. One pass, kept."""
        if reading in self._held:
            return self._held[reading]
        by_value: dict[Value, bytearray] = {}
        width = (self.count + 7) // 8
        for number, readings in enumerate(self._readings):
            value = readings.get(reading)
            bits = by_value.get(value)
            if bits is None:
                bits = by_value.setdefault(value, bytearray(width))
            bits[number >> 3] |= 1 << (number & 7)
        held = {value: int.from_bytes(bits, "little") for value, bits in by_value.items()}
        self._held[reading] = held
        return held

    def _against(self, first: str, second: str, same: bool) -> int:
        """The actions where one reading is the other, or is not — comparable ones only, since a comparison
        between a number and a name holds of nothing."""
        condition: Condition = (first, "== reading" if same else "!= reading", second)
        kept = self._compared.get(condition)
        if kept is not None:
            return kept
        width = (self.count + 7) // 8
        bits = bytearray(width)
        for number, readings in enumerate(self._readings):
            one, other = readings.get(first), readings.get(second)
            if isinstance(one, bool) != isinstance(other, bool):
                continue
            if isinstance(one, int | float) != isinstance(other, int | float):
                continue
            if (one == other) == same:
                bits[number >> 3] |= 1 << (number & 7)
        self._compared[condition] = int.from_bytes(bits, "little")
        return self._compared[condition]
