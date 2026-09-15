import textwrap

from openmind.agent.builder.domain_builder import DomainBuilder
from openmind.agent.model.domain import Domain
from openmind.csp.builder.problem_builder import ProblemBuilder
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rhetoric.constant.rhetoric_constant import (
    ACCOMMODATED,
    ASPECT,
    ASPECTS,
    DIRECTION,
    DIRECTIONS,
    DISTANCE_KINDS,
    EFFECTIVE_ANSWER,
    EFFECTIVE_IMPORTANCE,
    FORBIDDEN_NAME_CHARACTERS,
    KIND,
    MEMBER,
    MOVE,
    MOVES_LEFT,
    PAYOFF,
    PERCEIVED_ANSWER,
    PERCEIVED_IMPORTANCE,
    PERSUADED,
    PERSUADED_CHANCE,
    QUESTION,
    QUESTION_ID,
    SHOWN_ANSWER,
    STEP,
    TURN,
)
from openmind.rhetoric.model.rhetorical_scenario import RhetoricalScenario
from openmind.rule.model.python_rule import PythonRule
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State

#: What the effects of a move do, for either outcome of decreasing an audience distance.
EFFECTS = """\
if kind == 'identity':
    if aspect == 'distance':
        shift = toward if direction == 'decrease' else away
        shown_answer[member, question] = shift(shown_answer[member, question], effective_answer[question], STEP)
    else:
        effective_importance[question] = clamp(effective_importance[question] + signed(direction), 0.0, 1.0)
elif aspect == 'distance':
    if direction == 'increase':
        shown_answer[member, question] = away(shown_answer[member, question], perceived_answer[member, question], STEP)
    elif OUTCOME == 'persuaded':
        moved = STEP * (1.0 - perceived_importance[member, question])
        perceived_answer[member, question] = toward(perceived_answer[member, question], shown_answer[member, question], moved)
    else:
        shown_answer[member, question] = toward(shown_answer[member, question], perceived_answer[member, question], STEP)
else:
    perceived_importance[member, question] = clamp(perceived_importance[member, question] + signed(direction), 0.0, 1.0)
moves_left = moves_left - 1
if moves_left == 0:
    payoff = score(effective_answer, effective_importance, shown_answer, perceived_answer, perceived_importance)
"""


def create_rhetoric_domain(scenario: RhetoricalScenario) -> Domain:
    """The rhetorical game of a scenario, played by its speaker alone on what they know and perceive: each turn one
    strategic move on one audience member, question, kind and aspect, in one direction; the first reaction rules move
    answers and importances; after the scenario's moves, the payoff is how close the distances are to the goal.
    A member's name holding a comma or a parenthesis, or a speaker, member or goal naming a question the effective ethos
    doesn't answer, raises ValueError."""
    questions = _questions(scenario)
    members = tuple(member for member, _ in scenario.speaker.projective_ethos)
    definitions = _definitions(scenario, questions, members)
    return (
        DomainBuilder()
        .with_name(scenario.name)
        .with_initial_state(_initial_state(scenario, questions))
        .with_problem(_problem(questions, members, definitions))
        .with_transitions(_transitions(definitions))
        .with_players(Players((scenario.speaker.name,), TURN, (PAYOFF,)))
        .build()
    )


def _questions(scenario: RhetoricalScenario) -> dict[str, str]:
    speaker = scenario.speaker
    questions = {
        QUESTION_ID.format(number=number): position.question
        for number, position in enumerate(speaker.effective_ethos.positions, start=1)
    }
    known = set(questions.values())
    named = [
        *(position.question for _, ethos in speaker.projective_ethos for position in ethos.positions),
        *(position.question for _, pathos in speaker.projective_pathos for position in pathos.positions),
        *(target.question for target in scenario.goal.targets),
    ]
    if unknown := sorted(set(named) - known):
        raise ValueError(f"Questions the speaker's effective ethos doesn't answer: {unknown}")
    for member, _ in speaker.projective_ethos:
        if not member or any(character in member for character in FORBIDDEN_NAME_CHARACTERS):
            raise ValueError(f"A member's name can't be empty or hold {FORBIDDEN_NAME_CHARACTERS!r}: {member!r}")
    return questions


def _initial_state(scenario: RhetoricalScenario, questions: dict[str, str]) -> State:
    speaker, names, builder = scenario.speaker, VariableNameMapper(), StateBuilder()
    ids = {text: question for question, text in questions.items()}
    for position in speaker.effective_ethos.positions:
        question = ids[position.question]
        builder.with_variable(names.to_name(EFFECTIVE_ANSWER, (question,)), position.answer)
        builder.with_variable(names.to_name(EFFECTIVE_IMPORTANCE, (question,)), _importance(position.importance))
    perceived = dict(speaker.projective_pathos)
    for member, shown in speaker.projective_ethos:
        shown_answers = {position.question: position.answer for position in shown.positions}
        perceived_positions = {position.question: position for position in perceived[member].positions} if member in perceived else {}
        for question, text in questions.items():
            builder.with_variable(names.to_name(SHOWN_ANSWER, (member, question)), shown_answers.get(text, 0.0))
            position = perceived_positions.get(text)
            builder.with_variable(names.to_name(PERCEIVED_ANSWER, (member, question)), 0.0 if position is None else position.answer)
            builder.with_variable(
                names.to_name(PERCEIVED_IMPORTANCE, (member, question)),
                _importance(None if position is None else position.importance),
            )
    builder.with_variable(MOVES_LEFT, scenario.moves)
    builder.with_variable(TURN, speaker.name)
    builder.with_variable(PAYOFF, None)
    return builder.build()


def _problem(questions: dict[str, str], members: tuple[str, ...], definitions: PythonRule) -> Problem:
    variables = (
        Variable(MEMBER, DiscreteDomain(members)),
        Variable(QUESTION, DiscreteDomain(tuple(questions))),
        Variable(KIND, DiscreteDomain(DISTANCE_KINDS)),
        Variable(ASPECT, DiscreteDomain(ASPECTS)),
        Variable(DIRECTION, DiscreteDomain(DIRECTIONS)),
    )
    return (
        ProblemBuilder()
        .with_definitions(definitions)
        .with_action(MOVE, variables, (PythonRule(f"{PAYOFF} is None"),))
        .build()
    )


def _transitions(definitions: PythonRule) -> TransitionModel:
    persuaded = PythonRule(f"OUTCOME = {PERSUADED!r}\n{EFFECTS}")
    accommodated = PythonRule(f"OUTCOME = {ACCOMMODATED!r}\n{EFFECTS}")
    return (
        TransitionModelBuilder()
        .with_definitions(definitions)
        .with_transition(MOVE, (Branch(PERSUADED_CHANCE, persuaded), Branch(1.0 - PERSUADED_CHANCE, accommodated)))
        .build()
    )


def _definitions(scenario: RhetoricalScenario, questions: dict[str, str], members: tuple[str, ...]) -> PythonRule:
    header = textwrap.dedent(
        f"""\
        from openmind.rhetoric.model.distance_target import DistanceTarget
        from openmind.rhetoric.model.ethos import Ethos
        from openmind.rhetoric.model.pathos import Pathos
        from openmind.rhetoric.model.position import Position
        from openmind.rhetoric.model.rhetorical_goal import RhetoricalGoal
        from openmind.rhetoric.model.speaker import Speaker
        from openmind.rhetoric.service.distance_measurer import DistanceMeasurer
        from openmind.rhetoric.service.goal_scorer import GoalScorer

        SPEAKER = {scenario.speaker.name!r}
        QUESTIONS = {questions!r}
        MEMBERS = {members!r}
        GOAL = {scenario.goal!r}
        STEP = {STEP!r}
        """
    )
    body = textwrap.dedent(
        """\
        def clamp(value, low, high):
            return max(low, min(high, value))


        def signed(direction):
            return STEP if direction == 'increase' else -STEP


        def toward(value, target, step):
            if abs(target - value) <= step:
                return target
            return value + step if target > value else value - step


        def away(value, target, step):
            if target == value:
                return clamp(value - step if value > 0 else value + step, -1.0, 1.0)
            return clamp(value - step if target > value else value + step, -1.0, 1.0)


        def score(effective_answer, effective_importance, shown_answer, perceived_answer, perceived_importance):
            effective = Ethos(
                tuple(Position(text, effective_answer[question], effective_importance[question]) for question, text in QUESTIONS.items())
            )
            shown = tuple(
                (member, Ethos(tuple(Position(text, shown_answer[member, question]) for question, text in QUESTIONS.items())))
                for member in MEMBERS
            )
            perceived = tuple(
                (
                    member,
                    Pathos(
                        tuple(
                            Position(text, perceived_answer[member, question], perceived_importance[member, question])
                            for question, text in QUESTIONS.items()
                        )
                    ),
                )
                for member in MEMBERS
            )
            return GoalScorer().score(GOAL, DistanceMeasurer().distances(Speaker(SPEAKER, effective, shown, perceived)))
        """
    )
    return PythonRule(f"{header}\n{body}")


def _importance(importance: float | None) -> float:
    return 1.0 if importance is None else importance
