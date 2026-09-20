from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Bin:
    """A range of values an outcome can fall in, with how likely it is: what a continuous distribution is weighed by.

    A trip to the casino pays 0 to 1000 $. Binned, it reads: 0–100 $ isn't worth the trip, 100–200 $ is break-even,
    200–1000 $ is worth it. Each bin's value times its likelihood says whether to go."""

    low: float
    high: float
    likelihood: float
