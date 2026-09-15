#: Seconds a language model has to answer one rule.
DEFAULT_EXPLAINER_TIMEOUT = 300.0
#: Where explanations are cached by default, as <directory>/<domain>/<model>.json.
DEFAULT_EXPLANATIONS_DIRECTORY = "data/explanations"
#: What a language model is asked for each rule; filled with the domain, its variables, the rule's weight, source and
#: literal reading.
EXPLANATION_PROMPT = """You explain one rule of a program that values positions of the game "{domain}".
The rule is a Python expression. It reads the position through `here`. `me` is the player the position is valued \
for, and `other` is the opponent.

What a position holds: variables, with the values seen at the start of the game.
{variables}

What a position can do, for a position `v`:
- `v.best(player, lambda after: ...)` and `v.worst(player, lambda after: ...)`: the highest and the lowest value of \
the lambda over every move `player` could make, the lambda reading the position after that move;
- `v.count(player, lambda after: ...)`: how many of those moves the lambda holds after;
- `v.mobility(player)`: how many moves `player` could make;
- `v.offset(base, at, *steps)`: the variable of `base` at index `at` shifted by the steps, or '<outside>' off the \
positions that exist.
A comparison counts as 1 when true and 0 when false.

The rule's weight is {weight:+.4g}: a positive weight raises the player's value when the rule's number grows, a \
negative weight lowers it.
Python: {source}
Literal reading: {reading}

In one plain English sentence for a {domain} player, say what situation this rule measures and whether it is good or \
bad for the player. Say only what the rule computes; name the game's concepts when the rule clearly computes them. \
Answer with the sentence only."""
