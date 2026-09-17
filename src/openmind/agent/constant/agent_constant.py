import math

DEFAULT_ITERATIONS = 1000
EXPLORATION = math.sqrt(2)
PRIOR_WEIGHT = 1.0
ROLLOUT_TEMPERATURE = 0.2
GUIDED_ROLLOUTS = True
#: Each player's payoff for a rollout stopped at a rollout limit, by default: a draw in games paying 1, 0.5 and 0.
DEFAULT_UNFINISHED_PAYOFF = 0.5
#: The entry point group under which installed projects register their domains' recipes.
DOMAIN_ENTRY_POINTS = "openmind.domains"
#: The parameter a domain's timeout rule reads: the name of the player whose clock ran out.
FLAGGED = "flagged"
#: How many hex digits of a model's text's SHA-256 make its id.
MODEL_ID_DIGITS = 16
#: What a part of an agent that can't describe itself is written as, besides its class: it can't be built again from
#: the text.
NOT_REBUILDABLE = "not rebuildable"
#: The kinds of game remembered: self-play, games between arms, their held-out games, and matches between policies.
SELF_PLAY_GAME, HELD_OUT_SELF_PLAY_GAME, ARMS_GAME, HELD_OUT_ARMS_GAME, MATCH_GAME = (
    "self-play",
    "held-out self-play",
    "arms",
    "held-out arms",
    "match",
)
#: What a game gave a player: the highest payoff alone wins, a highest payoff shared draws, anything lower loses.
WIN, DRAW, LOSS = "win", "draw", "loss"
#: The keywords of the records a game leaves: each model once, and the game itself.
MODEL_KEYWORD, GAME_KEYWORD = "model", "game"
#: How a random policy is described.
RANDOM_POLICY_TEXT = '{"policy": "uniformly random legal actions"}'
#: The parameter a domain's picture rule reads: the action that led to the position, None at the start.
LAST_ACTION = "last"
