from importlib.metadata import entry_points

from openmind.agent.constant.agent_constant import DOMAIN_ENTRY_POINTS
from openmind.agent.constant.prisoners_dilemma_constant import NAME as PRISONERS_DILEMMA
from openmind.agent.constant.prisoners_dilemma_constant import STANDARD as PRISONERS_DILEMMA_STANDARD
from openmind.agent.constant.prisoners_dilemma_constant import VARIANTS as PRISONERS_DILEMMA_VARIANTS
from openmind.agent.constant.rock_paper_scissors_constant import NAME as ROCK_PAPER_SCISSORS
from openmind.agent.constant.sudoku_constant import NAME as SUDOKU
from openmind.agent.constant.tictactoe_constant import NAME as TICTACTOE
from openmind.agent.constant.tictactoe_constant import SEPARATOR, STANDARD, VARIANTS
from openmind.agent.factory.prisoners_dilemma_factory import declare_prisoners_dilemma
from openmind.agent.factory.rock_paper_scissors_factory import declare_rock_paper_scissors
from openmind.agent.factory.sudoku_factory import declare_sudoku
from openmind.agent.factory.tictactoe_factory import declare_tictactoe
from openmind.agent.model.game_rules import GameRules
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.rule_based_system import RuleBasedSystem


def declare_game(name: str, knowledge_base: KnowledgeBase) -> str:
    """Declares a game's rules into the knowledge base from its name and gives back the context they were declared
    under: "tictactoe", a variant such as "tictactoe/fourinarow", "sudoku", "prisonersdilemma", a variant such as
    "prisonersdilemma/uncertain", "rockpaperscissors", or a game an installed project registers under the
    openmind.domains entry points. An installed game is found by the part of the name before "/" and its rules get the
    whole name; the built-in games come first. Declaring twice leaves the knowledge base as it was."""
    game, separator, variant = name.partition(SEPARATOR)
    if game == TICTACTOE:
        if not separator:
            return declare_tictactoe(knowledge_base)
        if variant in VARIANTS:
            return declare_tictactoe(knowledge_base, VARIANTS[variant])
        raise ValueError(f"Unknown variant {variant!r} of {TICTACTOE}; variants: {', '.join(VARIANTS)}")
    if name == SUDOKU:
        return declare_sudoku(knowledge_base)
    if game == PRISONERS_DILEMMA:
        if not separator:
            return declare_prisoners_dilemma(knowledge_base)
        if variant in PRISONERS_DILEMMA_VARIANTS:
            return declare_prisoners_dilemma(knowledge_base, PRISONERS_DILEMMA_VARIANTS[variant])
        raise ValueError(
            f"Unknown variant {variant!r} of {PRISONERS_DILEMMA}; variants: {', '.join(PRISONERS_DILEMMA_VARIANTS)}"
        )
    if name == ROCK_PAPER_SCISSORS:
        return declare_rock_paper_scissors(knowledge_base)
    installed = {entry_point.name: entry_point for entry_point in entry_points(group=DOMAIN_ENTRY_POINTS)}
    if game in installed:
        rules: GameRules = installed[game].load()
        return rules(name, knowledge_base)
    variants = [SEPARATOR.join((TICTACTOE, variant)) for variant in VARIANTS if variant != STANDARD.name]
    dilemmas = [
        SEPARATOR.join((PRISONERS_DILEMMA, variant))
        for variant in PRISONERS_DILEMMA_VARIANTS
        if variant != PRISONERS_DILEMMA_STANDARD.name
    ]
    known = ", ".join((TICTACTOE, *variants, SUDOKU, PRISONERS_DILEMMA, *dilemmas, ROCK_PAPER_SCISSORS, *sorted(installed)))
    raise ValueError(f"Unknown game {name!r}; known games: {known}")


def create_game(name: str, knowledge_base: KnowledgeBase) -> RuleBasedSystem:
    """The RBS for a game, its rules declared into the knowledge base first where they aren't there yet."""
    return create_rule_based_system(knowledge_base, declare_game(name, knowledge_base))
