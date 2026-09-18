import textwrap

from openmind.agent.constant.prisoners_dilemma_constant import (
    CERTAIN,
    CHOICE,
    CHOICES,
    CHOOSE,
    CHOSEN,
    ENDING,
    NAME,
    PAYOFF,
    PLAYED,
    PLAYERS,
    POINTS,
    ROUND,
    SCORE,
    SEPARATOR,
    STANDARD,
    TURN,
    UNSET,
)
from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.model.python_rule import PythonRule
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.constant.players_constant import PLAYER
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State


def create_prisoners_dilemma_initial_state(variant: PrisonersDilemmaVariant = STANDARD) -> State:
    """Round 1, A to choose first, or both to choose at once in a simultaneous variant (turn(A) and turn(B) true),
    neither player's choice made nor played, both scores 0, no payoff set."""
    variable_name_mapper = VariableNameMapper()
    builder = StateBuilder()
    for player in PLAYERS:
        builder.with_variable(variable_name_mapper.to_name(CHOSEN, (player,)), UNSET)
        builder.with_variable(variable_name_mapper.to_name(PLAYED, (1, player)), UNSET)
        builder.with_variable(variable_name_mapper.to_name(SCORE, (player,)), 0)
        builder.with_variable(variable_name_mapper.to_name(PAYOFF, (player,)), UNSET)
        if variant.simultaneous:
            builder.with_variable(variable_name_mapper.to_name(TURN, (player,)), True)
    builder.with_variable(ROUND, 1)
    if not variant.simultaneous:
        builder.with_variable(TURN, PLAYERS[0])
    return builder.build()


def create_prisoners_dilemma_definitions(variant: PrisonersDilemmaVariant = STANDARD) -> PythonRule:
    """The names every rule of the variant sees: the PLAYERS, the CHOICES, POINTS[choice of A, choice of B], each
    player's points for a round, ROUNDS, the number of rounds or None without a known last round, other(player), and
    chosen_name(player), the name of a player's chosen variable."""
    _check(variant)
    return PythonRule(
        textwrap.dedent(
            f"""\
            PLAYERS = {PLAYERS!r}
            CHOICES = {CHOICES!r}
            POINTS = {POINTS!r}
            ROUNDS = {variant.rounds!r}


            def other(player):
                return PLAYERS[1] if player == PLAYERS[0] else PLAYERS[0]


            def chosen_name(player):
                return {CHOSEN!r} + '(' + player + ')'
            """
        )
    )


def declare_prisoners_dilemma_moves(declarer: RuleDeclarer, variant: PrisonersDilemmaVariant = STANDARD) -> None:
    """While no payoff is set, the player to act chooses to cooperate or defect, once a round; in a simultaneous variant,
    every player to act, read as `player`."""
    no_payoff_set = tuple(PythonRule(f"{PAYOFF}[{player!r}] is {UNSET!r}") for player in PLAYERS)
    if variant.simultaneous:
        constraints = (*no_payoff_set, PythonRule(f"{TURN}[{PLAYER}]"), PythonRule(f"{CHOSEN}[{PLAYER}] is {UNSET!r}"))
    else:
        constraints = (*no_payoff_set, PythonRule(f"{CHOSEN}[{TURN}] is {UNSET!r}"))
    declarer.values(CHOOSE, CHOICE, PythonRule(repr(CHOICES)))
    declarer.constraints(CHOOSE, *constraints)


def declare_prisoners_dilemma_effects(declarer: RuleDeclarer, variant: PrisonersDilemmaVariant = STANDARD) -> None:
    """Keep the choice until both players have chosen; then play both, add their points, and either end the game, each
    payoff the player's own score, or start the next round with its choices not yet played. The game ends after the
    variant's last round, and with the variant's ending chance after any round: the action then has a branch where the
    game goes on and one where it ends, which give the same state when the choice doesn't finish a round. In a
    simultaneous variant, each choice only keeps its player's choice, and the resolution, run once both have chosen,
    plays the round with those same branches, ending the game by clearing both players' turns."""
    _check(variant)
    declarer.definitions(create_prisoners_dilemma_definitions(variant), effects=True)
    if variant.simultaneous:
        _declare_simultaneous(declarer, variant)
        return
    effects = textwrap.dedent(
        f"""\
        {CHOSEN}[{TURN}] = {CHOICE}
        if all({CHOSEN}[player] is not {UNSET!r} for player in PLAYERS):
            points = POINTS[{CHOSEN}[PLAYERS[0]], {CHOSEN}[PLAYERS[1]]]
            for player, gained in zip(PLAYERS, points):
                {PLAYED}[{ROUND}, player] = {CHOSEN}[player]
                {SCORE}[player] = {SCORE}[player] + gained
                {CHOSEN}[player] = {UNSET!r}
            if {ROUND} == ROUNDS or {ENDING}:
                for player in PLAYERS:
                    {PAYOFF}[player] = {SCORE}[player]
            else:
                {ROUND} = {ROUND} + 1
                for player in PLAYERS:
                    {PLAYED}[{ROUND}, player] = {UNSET!r}
        {TURN} = other({TURN})
        """
    )
    goes_on, ends = (PythonRule(f"{ENDING} = {ending!r}\n{effects}") for ending in (False, True))
    for number, (rule, chance) in enumerate(_outcomes(variant, goes_on, ends), start=1):
        declarer.leads_to(CHOOSE, rule, chance, None if variant.ending_chance in (0.0, 1.0) else number)


def create_prisoners_dilemma_players() -> Players:
    """A and B; turn names the player to act; payoff(A) and payoff(B) hold their payoffs."""
    variable_name_mapper = VariableNameMapper()
    return Players(PLAYERS, TURN, tuple(variable_name_mapper.to_name(PAYOFF, (player,)) for player in PLAYERS))


def declare_prisoners_dilemma(
    knowledge_base: KnowledgeBase, variant: PrisonersDilemmaVariant = STANDARD, weight: float = 1.0
) -> str:
    """Declares the repeated prisoner's dilemma's rules, or one of its variants', and gives back the context they were
    declared under: "prisonersdilemma" for the standard game, "prisonersdilemma/<variant>" for any other. A variant with
    fewer than 1 round, an ending chance outside 0 to 1, or neither a last round nor an ending chance raises ValueError.
    In a simultaneous variant a choice is kept only until the round's resolution, which no player sees happen before
    choosing."""
    context = NAME if variant == STANDARD else SEPARATOR.join((NAME, variant.name))
    declarer = RuleDeclarer(knowledge_base, context, weight)
    declarer.starts_at(create_prisoners_dilemma_initial_state(variant))
    declarer.played_by(create_prisoners_dilemma_players())
    declarer.definitions(create_prisoners_dilemma_definitions(variant))
    declare_prisoners_dilemma_moves(declarer, variant)
    declare_prisoners_dilemma_effects(declarer, variant)
    return declarer.done()


def _outcomes(
    variant: PrisonersDilemmaVariant, goes_on: PythonRule, ends: PythonRule
) -> tuple[tuple[PythonRule, float], ...]:
    """The outcomes of a round with their chances: the game goes on, it ends, or either with the variant's chance."""
    if variant.ending_chance == 0.0:
        return ((goes_on, CERTAIN),)
    if variant.ending_chance == 1.0:
        return ((ends, CERTAIN),)
    return ((goes_on, 1.0 - variant.ending_chance), (ends, variant.ending_chance))


def _declare_simultaneous(declarer: RuleDeclarer, variant: PrisonersDilemmaVariant) -> None:
    resolution = textwrap.dedent(
        f"""\
        points = POINTS[{CHOSEN}[PLAYERS[0]], {CHOSEN}[PLAYERS[1]]]
        for each, gained in zip(PLAYERS, points):
            {PLAYED}[{ROUND}, each] = {CHOSEN}[each]
            {SCORE}[each] = {SCORE}[each] + gained
            {CHOSEN}[each] = {UNSET!r}
        if {ROUND} == ROUNDS or {ENDING}:
            for each in PLAYERS:
                {PAYOFF}[each] = {SCORE}[each]
                {TURN}[each] = False
        else:
            {ROUND} = {ROUND} + 1
            for each in PLAYERS:
                {PLAYED}[{ROUND}, each] = {UNSET!r}
        """
    )
    goes_on, ends = (PythonRule(f"{ENDING} = {ending!r}\n{resolution}") for ending in (False, True))
    declarer.leads_to(CHOOSE, PythonRule(f"{CHOSEN}[{PLAYER}] = {CHOICE}"))
    for number, (rule, chance) in enumerate(_outcomes(variant, goes_on, ends), start=1):
        declarer.together(rule, chance, None if variant.ending_chance in (0.0, 1.0) else number)


def _check(variant: PrisonersDilemmaVariant) -> None:
    rounds, chance = variant.rounds, variant.ending_chance
    if (rounds is not None and rounds < 1) or not 0.0 <= chance <= 1.0 or (rounds is None and chance == 0.0):
        raise ValueError(
            "A prisoner's dilemma variant needs at least 1 round, an ending chance from 0 to 1, and an ending chance "
            f"above 0 when no last round is known, not {variant!r}"
        )
