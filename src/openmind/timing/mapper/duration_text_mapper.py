#: How many seconds are in a minute, an hour and a day, for saying a length of time the way a person says it.
SECONDS_A_MINUTE = 60.0
SECONDS_AN_HOUR = SECONDS_A_MINUTE * 60
SECONDS_A_DAY = SECONDS_AN_HOUR * 24


class DurationTextMapper:
    """A length of time as a person says it.

    Seconds are what a clock holds and what a budget is spent in, and they stop being readable somewhere around a
    hundred: nobody reads 4916 seconds as an hour and twenty minutes without doing the arithmetic. Anything a person
    is meant to read — a log of how long a game took, how long a run has been going — says it in the largest units
    it fills, and keeps the next one down so it stays exact enough to compare."""

    def to_text(self, seconds: float) -> str:
        """That many seconds in words: `4916` as `1 hour 21 minutes`, `587` as `9 minutes 47 seconds`, `0.2` as
        `0.2 seconds`."""
        if seconds < 0:
            return f"minus {self.to_text(-seconds)}"
        seconds = self._rounded(seconds)
        if seconds < 10:
            return f"{seconds:.3g} seconds" if seconds != 1 else "1 second"
        if seconds < SECONDS_A_MINUTE:
            return f"{round(seconds)} seconds"
        if seconds < SECONDS_AN_HOUR:
            return self._said(seconds, SECONDS_A_MINUTE, "minute", "second")
        if seconds < SECONDS_A_DAY:
            return self._said(seconds, SECONDS_AN_HOUR, "hour", "minute")
        return self._said(seconds, SECONDS_A_DAY, "day", "hour")

    def _rounded(self, seconds: float) -> float:
        """The length rounded to what will be shown of it, before choosing how to say it.

        Said without rounding first, a second under the hour comes out as sixty minutes: the choice of words is made
        on the unrounded length and the rounding then fills the unit it was about to leave."""
        if seconds < 10:
            return seconds
        if seconds < SECONDS_AN_HOUR:
            return float(round(seconds))
        if seconds < SECONDS_A_DAY:
            return round(seconds / SECONDS_A_MINUTE) * SECONDS_A_MINUTE
        return round(seconds / SECONDS_AN_HOUR) * SECONDS_AN_HOUR

    def _said(self, seconds: float, unit: float, name: str, smaller: str) -> str:
        """So many of that unit and so many of the next one down, leaving out what is zero."""
        smaller_unit = {"minute": 1.0, "hour": SECONDS_A_MINUTE, "day": SECONDS_AN_HOUR}[name]
        many = int(seconds // unit)
        left = round((seconds - many * unit) / smaller_unit)
        if left * smaller_unit >= unit:
            many, left = many + 1, 0
        return f"{self._many(many, name)} {self._many(left, smaller)}" if left else self._many(many, name)

    def _many(self, many: int, name: str) -> str:
        return f"{many} {name}" if many == 1 else f"{many} {name}s"
