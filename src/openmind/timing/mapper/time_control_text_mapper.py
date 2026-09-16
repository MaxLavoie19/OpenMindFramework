import re

from openmind.timing.model.time_control import TimeControl

#: A time control as chess writes one: the base in minutes, a plus, the increment in seconds.
TEXT = re.compile(r"^\s*(?P<minutes>\d+(?:\.\d+)?)\s*\+\s*(?P<seconds>\d+(?:\.\d+)?)\s*$")
SECONDS_A_MINUTE = 60.0


class TimeControlTextMapper:
    """A time control as chess writes one, base minutes plus increment seconds: `3+2` is three minutes and two seconds a
    move, `1+0` a minute with nothing added, `0.5+1` thirty seconds and one a move."""

    def from_text(self, text: str) -> TimeControl:
        """The time control the text writes; anything else, or a base of zero, raises ValueError."""
        found = TEXT.match(text)
        if found is None:
            raise ValueError(f"A time control is written as minutes+seconds, such as 3+2, not {text!r}")
        return TimeControl(float(found["minutes"]) * SECONDS_A_MINUTE, float(found["seconds"]))

    def to_text(self, control: TimeControl) -> str:
        return f"{control.base_seconds / SECONDS_A_MINUTE:g}+{control.increment_seconds:g}"
