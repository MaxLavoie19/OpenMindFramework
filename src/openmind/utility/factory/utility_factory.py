from openmind.utility.service.even_binner import EvenBinner
from openmind.utility.service.utility import Utility


def create_utility() -> Utility:
    """What a move is worth over its outcomes, goals and preferences."""
    return Utility()


def create_even_binner(bins: int = 3) -> EvenBinner:
    """The binning model cutting a range of payoffs into bins of equal width."""
    return EvenBinner(bins)
