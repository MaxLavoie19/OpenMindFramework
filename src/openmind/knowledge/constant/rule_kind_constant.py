#: The state the game starts in.
INITIAL = "initial"

#: Who plays, in what order, the variable naming the player to act and the payoff variables.
PLAYERS = "players"

#: What a grid's cell holds when nothing is on it, its base named as the rule's parameter.
EMPTY = "empty"

#: The definitions script a context's rules all see: every name it leaves is theirs.
DEFINITIONS = "definitions"

#: Whether an action with these parameter values is legal.
CONSTRAINT = "constraint"

#: The values a parameter of an action can take.
VALUES = "values"

#: What an action leads to.
EFFECTS = "effects"

#: Why a finished game ended.
ENDING = "ending"

#: A game's record, from its initial state and the actions played.
RECORD = "record"

#: What a player's clock running out does to the payoffs.
TIMEOUT = "timeout"

#: A position as an image, for pages showing games.
PICTURE = "picture"

#: A position's value for a player: a position heuristic's term.
POSITION = "position"

#: A move's value for the player to act: a move heuristic's term.
MOVE = "move"

#: What a game project registers: the rules of the game itself.
GAME_KINDS = (INITIAL, PLAYERS, EMPTY, DEFINITIONS, CONSTRAINT, VALUES, EFFECTS, ENDING, RECORD, TIMEOUT, PICTURE)

#: What the inference engine and fitting produce: rules that judge rather than rule.
HEURISTIC_KINDS = (POSITION, MOVE)

#: Every kind of rule the knowledge base holds.
RULE_KINDS = GAME_KINDS + HEURISTIC_KINDS

