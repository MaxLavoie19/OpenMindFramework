from importlib.metadata import entry_points

from openmind.agent.constant.agent_constant import DOMAIN_ENTRY_POINTS
from openmind.agent.constant.sudoku_constant import NAME as SUDOKU
from openmind.agent.constant.tictactoe_constant import NAME as TICTACTOE
from openmind.agent.constant.tictactoe_constant import SEPARATOR, STANDARD, VARIANTS
from openmind.agent.factory.sudoku_factory import create_sudoku_domain
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.model.domain import Domain
from openmind.agent.model.domain_recipe import DomainRecipe


def create_domain(name: str) -> Domain:
    """Creates a domain within the agent from its name: "tictactoe", a variant such as "tictactoe/fourinarow", "sudoku",
    or a domain an installed project registers under the openmind.domains entry points. An installed domain is found
    by the part of the name before "/" and its recipe gets the whole name; the built-in domains come first."""
    game, separator, variant = name.partition(SEPARATOR)
    if game == TICTACTOE:
        if not separator:
            return create_tictactoe_domain()
        if variant in VARIANTS:
            return create_tictactoe_domain(VARIANTS[variant])
        raise ValueError(f"Unknown variant {variant!r} of {TICTACTOE}; variants: {', '.join(VARIANTS)}")
    if name == SUDOKU:
        return create_sudoku_domain()
    installed = {entry_point.name: entry_point for entry_point in entry_points(group=DOMAIN_ENTRY_POINTS)}
    if game in installed:
        recipe: DomainRecipe = installed[game].load()
        return recipe(name)
    variants = [SEPARATOR.join((TICTACTOE, variant)) for variant in VARIANTS if variant != STANDARD.name]
    known = ", ".join((TICTACTOE, *variants, SUDOKU, *sorted(installed)))
    raise ValueError(f"Unknown domain {name!r}; known domains: {known}")
