#: The state the game starts in.
INITIAL = "initial"

#: Who plays, and the model holding each player's payoff.
PLAYERS = "players"

#: The definitions script a context's rules all see: every name it leaves is theirs.
DEFINITIONS = "definitions"

#: Whether an action with these parameter values is legal.
CONSTRAINT = "constraint"

#: The values a parameter of an action can take.
VALUES = "values"

#: Every legal action of a player in a state, listed outright.
#:
#: The constraints are what makes an action legal; this is a game saying it can list the same actions faster than
#: they can be solved for. Where it is declared it is used and the constraints are not, so the two must agree — a
#: game whose listing and whose constraints say different things is a game with a bug in it, and OMF has no way to
#: know which of the two it meant. It is dropped wherever the constraints are the point: a relaxation is the game
#: with a constraint taken away, and a listing that still lists every legal action would make the relaxation no
#: relaxation at all.
LISTING = "listing"

#: What an action leads to.
EFFECTS = "effects"

#: Why a finished game ended.
ENDING = "ending"

#: A game's record, from its initial state and the actions played.
RECORD = "record"

#: How long an action takes to perform, in seconds, from the state and its parameters.
DURATION = "duration"

#: How long before an action is available again, in seconds, from the state and its parameters.
COOLDOWN = "cooldown"

#: A position as an image, for pages showing games.
PICTURE = "picture"

#: The action a policy settles on in a state: an optimizer's rule.
OPTIMUM = "optimum"

#: A position's value for a player: a position heuristic's term.
POSITION = "position"

#: A move's value for the player to act: a move heuristic's term.
MOVE = "move"

#: What one level of the hierarchy sees of a state: an abstraction model's rule.
ABSTRACTION = "abstraction"

#: What a game project registers: the rules of the game itself.
GAME_KINDS = (
    INITIAL,
    PLAYERS,
    DEFINITIONS,
    CONSTRAINT,
    VALUES,
    LISTING,
    EFFECTS,
    ENDING,
    DURATION,
    COOLDOWN,
    RECORD,
    PICTURE,
)

#: What the inference engine and fitting produce: rules that judge rather than rule.
HEURISTIC_KINDS = (POSITION, MOVE, OPTIMUM)

#: Every kind of rule the knowledge base holds.
RULE_KINDS = GAME_KINDS + HEURISTIC_KINDS + (ABSTRACTION,)

