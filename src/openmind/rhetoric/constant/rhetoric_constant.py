#: The distance between the speaker's effective ethos and the ethos they project to a member: affirming or hiding
#: their identity.
IDENTITY = "identity"
#: The distance between the ethos a speaker projects to a member and that member's pathos as the speaker perceives it.
AUDIENCE = "audience"
DISTANCE_KINDS = (IDENTITY, AUDIENCE)
#: What a strategic move changes, and in which direction.
DISTANCE = "distance"
PROBLEMATICITY = "problematicity"
ASPECTS = (DISTANCE, PROBLEMATICITY)
INCREASE = "increase"
DECREASE = "decrease"
DIRECTIONS = (INCREASE, DECREASE)
#: A position a message says: an answer of 1 is yes and -1 is no.
YES = 1.0
NO = -1.0

#: The rhetorical game's state variables: the speaker's effective positions and importances by question, the answers
#: shown to each member, and each member's perceived answers and importances; the moves left; the player to act; the
#: payoff.
EFFECTIVE_ANSWER = "effective_answer"
EFFECTIVE_IMPORTANCE = "effective_importance"
SHOWN_ANSWER = "shown_answer"
PERCEIVED_ANSWER = "perceived_answer"
PERCEIVED_IMPORTANCE = "perceived_importance"
MOVES_LEFT = "moves_left"
TURN = "turn"
PAYOFF = "payoff"
#: The strategic move action and its parameters.
MOVE = "move"
MEMBER = "member"
QUESTION = "question"
KIND = "kind"
ASPECT = "aspect"
DIRECTION = "direction"
#: How a question is named in the state: q1, q2, ... in the order the speaker's effective ethos answers them.
QUESTION_ID = "q{number}"
#: Characters a member's name can't hold, since names index state variables.
FORBIDDEN_NAME_CHARACTERS = "(),"

#: The first, hand-written reaction rules. A move shifts an answer by STEP toward or away from the other position, or an
#: importance by STEP, staying within -1 to 1 and 0 to 1. Decreasing the audience distance has two outcomes: with
#: PERSUADED_CHANCE the member is persuaded, their answer moving toward the shown one by STEP times one minus how much
#: the question matters to them; otherwise the speaker accommodates, their shown answer moving toward the member's.
STEP = 0.5
PERSUADED_CHANCE = 0.5
PERSUADED = "persuaded"
ACCOMMODATED = "accommodated"
