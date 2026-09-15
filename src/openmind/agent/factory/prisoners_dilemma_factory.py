import textwrap

from openmind.agent.builder.domain_builder import DomainBuilder
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
from openmind.agent.model.domain import Domain
from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant
from openmind.csp.builder.problem_builder import ProblemBuilder
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.observation.constant.observation_constant import PLAYER
from openmind.observation.model.observation import Observation
from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.world.builder.state_builder import StateBuilder
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


def create_prisoners_dilemma_problem(variant: PrisonersDilemmaVariant = STANDARD) -> Problem:
    """While no payoff is set, the player to act chooses to cooperate or defect, once a round; in a simultaneous variant,
    every player to act, read as `player`."""
    no_payoff_set = tuple(PythonRule(f"{PAYOFF}[{player!r}] is {UNSET!r}") for player in PLAYERS)
    if variant.simultaneous:
        constraints = (*no_payoff_set, PythonRule(f"{TURN}[{PLAYER}]"), PythonRule(f"{CHOSEN}[{PLAYER}] is {UNSET!r}"))
    else:
        constraints = (*no_payoff_set, PythonRule(f"{CHOSEN}[{TURN}] is {UNSET!r}"))
    return ProblemBuilder().with_action(CHOOSE, (Variable(CHOICE, DiscreteDomain(CHOICES)),), constraints).build()


def create_prisoners_dilemma_transitions(variant: PrisonersDilemmaVariant = STANDARD) -> TransitionModel:
    """Keep the choice until both players have chosen; then play both, add their points, and either end the game, each
    payoff the player's own score, or start the next round with its choices not yet played. The game ends after the
    variant's last round, and with the variant's ending chance after any round: the action then has a branch where the
    game goes on and one where it ends, which give the same state when the choice doesn't finish a round. In a
    simultaneous variant, each choice only keeps its player's choice, and the resolution, run once both have chosen,
    plays the round with those same branches, ending the game by clearing both players' turns."""
    _check(variant)
    if variant.simultaneous:
        return _simultaneous_transitions(variant)
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
    if variant.ending_chance == 0.0:
        branches = (Branch(CERTAIN, goes_on),)
    elif variant.ending_chance == 1.0:
        branches = (Branch(CERTAIN, ends),)
    else:
        branches = (Branch(1.0 - variant.ending_chance, goes_on), Branch(variant.ending_chance, ends))
    return (
        TransitionModelBuilder()
        .with_definitions(create_prisoners_dilemma_definitions(variant))
        .with_transition(CHOOSE, branches)
        .build()
    )


def create_prisoners_dilemma_observation(variant: PrisonersDilemmaVariant = STANDARD) -> Observation:
    """A player can't see the other player's choice: the other's chosen variable is hidden. What it could be: either
    choice at even chances when the other has already chosen this round, which is B's case on B's turn, and None
    otherwise."""
    return Observation(
        PythonRule(f"(chosen_name(other({PLAYER})),)"),
        PythonRule(
            f"[({{chosen_name(other({PLAYER})): choice}}, 1 / len(CHOICES)) for choice in CHOICES] "
            f"if {PLAYER} == {TURN} == PLAYERS[1] else [({{chosen_name(other({PLAYER})): {UNSET!r}}}, 1.0)]"
        ),
        create_prisoners_dilemma_definitions(variant),
    )


def create_prisoners_dilemma_players() -> Players:
    """A and B; turn names the player to act; payoff(A) and payoff(B) hold their payoffs."""
    variable_name_mapper = VariableNameMapper()
    return Players(PLAYERS, TURN, tuple(variable_name_mapper.to_name(PAYOFF, (player,)) for player in PLAYERS))


def create_prisoners_dilemma_domain(variant: PrisonersDilemmaVariant = STANDARD) -> Domain:
    """The repeated prisoner's dilemma or one of its variants. The standard game is named "prisonersdilemma", any other
    variant "prisonersdilemma/<variant>". A variant with fewer than 1 round, an ending chance outside 0 to 1, or neither
    a last round nor an ending chance raises ValueError. A simultaneous variant has no observation: a choice is kept
    only until the round's resolution, which no player sees happen before choosing."""
    builder = (
        DomainBuilder()
        .with_name(NAME if variant == STANDARD else SEPARATOR.join((NAME, variant.name)))
        .with_initial_state(create_prisoners_dilemma_initial_state(variant))
        .with_problem(create_prisoners_dilemma_problem(variant))
        .with_transitions(create_prisoners_dilemma_transitions(variant))
        .with_players(create_prisoners_dilemma_players())
    )
    if not variant.simultaneous:
        builder.with_observation(create_prisoners_dilemma_observation(variant))
    return builder.build()


def _simultaneous_transitions(variant: PrisonersDilemmaVariant) -> TransitionModel:
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
    if variant.ending_chance == 0.0:
        branches = (Branch(CERTAIN, goes_on),)
    elif variant.ending_chance == 1.0:
        branches = (Branch(CERTAIN, ends),)
    else:
        branches = (Branch(1.0 - variant.ending_chance, goes_on), Branch(variant.ending_chance, ends))
    return (
        TransitionModelBuilder()
        .with_definitions(create_prisoners_dilemma_definitions(variant))
        .with_transition(CHOOSE, (Branch(CERTAIN, PythonRule(f"{CHOSEN}[{PLAYER}] = {CHOICE}")),))
        .with_resolution(branches)
        .build()
    )


def _check(variant: PrisonersDilemmaVariant) -> None:
    rounds, chance = variant.rounds, variant.ending_chance
    if (rounds is not None and rounds < 1) or not 0.0 <= chance <= 1.0 or (rounds is None and chance == 0.0):
        raise ValueError(
            "A prisoner's dilemma variant needs at least 1 round, an ending chance from 0 to 1, and an ending chance "
            f"above 0 when no last round is known, not {variant!r}"
        )
