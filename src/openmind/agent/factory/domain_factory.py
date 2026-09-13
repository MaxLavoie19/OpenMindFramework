from openmind.agent.constant.sudoku_constant import NAME as SUDOKU
from openmind.agent.constant.tictactoe_constant import NAME as TICTACTOE
from openmind.agent.constant.tictactoe_constant import SEPARATOR, STANDARD, VARIANTS
from openmind.agent.factory.sudoku_factory import create_sudoku_domain
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.model.domain import Domain


def create_domain(name: str) -> Domain:
    """Creates a domain within the agent from its name: "tictactoe", a variant such as "tictactoe/fourinarow", or
    "sudoku"."""
    game, separator, variant = name.partition(SEPARATOR)
    if game == TICTACTOE:
        if not separator:
            return create_tictactoe_domain()
        if variant in VARIANTS:
            return create_tictactoe_domain(VARIANTS[variant])
        raise ValueError(f"Unknown variant {variant!r} of {TICTACTOE}; variants: {', '.join(VARIANTS)}")
    if name == SUDOKU:
        return create_sudoku_domain()
    variants = ", ".join(SEPARATOR.join((TICTACTOE, variant)) for variant in VARIANTS if variant != STANDARD.name)
    raise ValueError(f"Unknown domain {name!r}; known domains: {TICTACTOE}, {variants}, {SUDOKU}")
