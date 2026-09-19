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
    VARIANTS,
)
from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant
from openmind.game.service.game_declarer import GameDeclarer
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.constant.players_constant import PLAYER
from openmind.world.model.players import Players
from openmind.world.model.state import State


def create_prisoners_dilemma_initial_state(variant: PrisonersDilemmaVariant = STANDARD) -> State:
    """Round 1; A to choose first, except in a simultaneous variant, where both choose at once and there is no turn;
    the chosen map holding neither player's choice; the played grid, one row per round and one column per player
    in the players' order, holding round 1 not yet played; the score map holding 0 for both; the payoff map holding no
    payoff."""
    builder = (
        StateBuilder()
        .with_model(CHOSEN, Map.of(dict.fromkeys(PLAYERS, UNSET)))
        .with_model(PLAYED, Grid.filled((1, len(PLAYERS)), UNSET))
        .with_model(SCORE, Map.of(dict.fromkeys(PLAYERS, 0)))
        .with_model(PAYOFF, Map.of(dict.fromkeys(PLAYERS, UNSET)))
        .with_model(ROUND, 1)
    )
    return builder.build() if variant.simultaneous else builder.with_model(TURN, PLAYERS[0]).build()


def create_prisoners_dilemma_definitions(variant: PrisonersDilemmaVariant = STANDARD) -> PythonRule:
    """The names every rule of the variant sees: the PLAYERS, the CHOICES, POINTS[choice of A, choice of B], each
    player's points for a round, ROUNDS, the number of rounds or None without a known last round, and other(player)."""
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
            """
        )
    )


def declare_prisoners_dilemma_moves(declarer: GameDeclarer, variant: PrisonersDilemmaVariant = STANDARD) -> None:
    """While no payoff is set, a player who hasn't chosen this round chooses to cooperate or defect: the player whose
    turn it is, or, in a simultaneous variant, both at once."""
    no_payoff_set = tuple(PythonRule(f"{PAYOFF}[{player!r}] is {UNSET!r}") for player in PLAYERS)
    constraints = (*no_payoff_set, PythonRule(f"{CHOSEN}[{PLAYER}] is {UNSET!r}"))
    if not variant.simultaneous:
        constraints = (PythonRule(f"{TURN} == {PLAYER}"), *constraints)
    declarer.values(CHOOSE, CHOICE, PythonRule(repr(CHOICES)))
    declarer.constraints(CHOOSE, *constraints)


def declare_prisoners_dilemma_effects(declarer: GameDeclarer, variant: PrisonersDilemmaVariant = STANDARD) -> None:
    """Keep the choice until both players have chosen; then play both, add their points, and either end the game, each
    payoff the player's own score, or start the next round with its choices not yet played. The game ends after the
    variant's last round, and with the variant's ending chance after any round: the action then has a branch where the
    game goes on and one where it ends, which give the same state when the choice doesn't finish a round. In a
    simultaneous variant, each choice only keeps its player's choice, and the resolution, run once both have chosen,
    plays the round with those same branches."""
    _check(variant)
    declarer.definitions(create_prisoners_dilemma_definitions(variant), effects=True)
    if variant.simultaneous:
        _declare_simultaneous(declarer, variant)
        return
    effects = textwrap.dedent(
        f"""\
        {CHOSEN} = {CHOSEN}.with_item({TURN}, {CHOICE})
        if all({CHOSEN}[player] is not {UNSET!r} for player in PLAYERS):
            points = POINTS[{CHOSEN}[PLAYERS[0]], {CHOSEN}[PLAYERS[1]]]
            for column, (player, gained) in enumerate(zip(PLAYERS, points), start=1):
                {PLAYED} = {PLAYED}.placed(({ROUND}, column), {CHOSEN}[player])
                {SCORE} = {SCORE}.with_item(player, {SCORE}[player] + gained)
                {CHOSEN} = {CHOSEN}.with_item(player, {UNSET!r})
            if {ROUND} == ROUNDS or {ENDING}:
                for player in PLAYERS:
                    {PAYOFF} = {PAYOFF}.with_item(player, {SCORE}[player])
            else:
                {ROUND} = {ROUND} + 1
                {PLAYED} = Grid(({ROUND}, len(PLAYERS)), {PLAYED}.cells + ({UNSET!r},) * len(PLAYERS))
        {TURN} = other({TURN})
        """
    )
    goes_on, ends = (PythonRule(f"{ENDING} = {ending!r}\n{effects}") for ending in (False, True))
    for number, (rule, chance) in enumerate(_outcomes(variant, goes_on, ends), start=1):
        declarer.leads_to(CHOOSE, rule, chance, None if variant.ending_chance in (0.0, 1.0) else number)


def create_prisoners_dilemma_players() -> Players:
    """A and B; the payoff map holds each one's payoff."""
    return Players(PLAYERS, PAYOFF)


def declare_prisoners_dilemma(knowledge_base: KnowledgeBase, variant: PrisonersDilemmaVariant = STANDARD) -> str:
    """Declares the repeated prisoner's dilemma's rules, or one of its variants', and gives back the context they were
    declared under: "prisonersdilemma" for the standard game, "prisonersdilemma/<variant>" for any other. A variant with
    fewer than 1 round, an ending chance outside 0 to 1, or neither a last round nor an ending chance raises ValueError.
    In a simultaneous variant a choice is kept only until the round's resolution, which no player sees happen before
    choosing."""
    context = NAME if variant == STANDARD else SEPARATOR.join((NAME, variant.name))
    declarer = GameDeclarer(knowledge_base, context)
    declarer.starts_at(create_prisoners_dilemma_initial_state(variant))
    declarer.played_by(create_prisoners_dilemma_players())
    declarer.definitions(create_prisoners_dilemma_definitions(variant))
    declare_prisoners_dilemma_moves(declarer, variant)
    declare_prisoners_dilemma_effects(declarer, variant)
    return declarer.done()



def declare_prisoners_dilemma_named(name: str, knowledge_base: KnowledgeBase) -> str:
    """Declares the prisoner's dilemma from its registered name: "prisonersdilemma", or "prisonersdilemma/<variant>";
    an unknown variant raises ValueError."""
    _, separator, variant = name.partition(SEPARATOR)
    if not separator:
        return declare_prisoners_dilemma(knowledge_base)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r} of {NAME}; variants: {', '.join(VARIANTS)}")
    return declare_prisoners_dilemma(knowledge_base, VARIANTS[variant])

def _outcomes(
    variant: PrisonersDilemmaVariant, goes_on: PythonRule, ends: PythonRule
) -> tuple[tuple[PythonRule, float], ...]:
    """The outcomes of a round with their chances: the game goes on, it ends, or either with the variant's chance."""
    if variant.ending_chance == 0.0:
        return ((goes_on, CERTAIN),)
    if variant.ending_chance == 1.0:
        return ((ends, CERTAIN),)
    return ((goes_on, 1.0 - variant.ending_chance), (ends, variant.ending_chance))


def _declare_simultaneous(declarer: GameDeclarer, variant: PrisonersDilemmaVariant) -> None:
    resolution = textwrap.dedent(
        f"""\
        points = POINTS[{CHOSEN}[PLAYERS[0]], {CHOSEN}[PLAYERS[1]]]
        for column, (each, gained) in enumerate(zip(PLAYERS, points), start=1):
            {PLAYED} = {PLAYED}.placed(({ROUND}, column), {CHOSEN}[each])
            {SCORE} = {SCORE}.with_item(each, {SCORE}[each] + gained)
            {CHOSEN} = {CHOSEN}.with_item(each, {UNSET!r})
        if {ROUND} == ROUNDS or {ENDING}:
            for each in PLAYERS:
                {PAYOFF} = {PAYOFF}.with_item(each, {SCORE}[each])
        else:
            {ROUND} = {ROUND} + 1
            {PLAYED} = Grid(({ROUND}, len(PLAYERS)), {PLAYED}.cells + ({UNSET!r},) * len(PLAYERS))
        """
    )
    goes_on, ends = (PythonRule(f"{ENDING} = {ending!r}\n{resolution}") for ending in (False, True))
    declarer.leads_to(CHOOSE, PythonRule(f"{CHOSEN} = {CHOSEN}.with_item({PLAYER}, {CHOICE})"))
    for number, (rule, chance) in enumerate(_outcomes(variant, goes_on, ends), start=1):
        declarer.together(rule, chance, None if variant.ending_chance in (0.0, 1.0) else number)


def _check(variant: PrisonersDilemmaVariant) -> None:
    rounds, chance = variant.rounds, variant.ending_chance
    if (rounds is not None and rounds < 1) or not 0.0 <= chance <= 1.0 or (rounds is None and chance == 0.0):
        raise ValueError(
            "A prisoner's dilemma variant needs at least 1 round, an ending chance from 0 to 1, and an ending chance "
            f"above 0 when no last round is known, not {variant!r}"
        )
