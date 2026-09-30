DEFAULT_SOLUTION_LIMIT = 2

#: Try a variable's values in the order its values rule gave them.
DOMAIN_ORDER = "domain order"

#: Try the value that rules out the fewest values elsewhere first. It finds one solution sooner where the search
#: has to guess, and costs a pass over what each value would rule out — which is worth it for a game's moves and
#: is not worth it where a variable has hundreds of values and the search never backtracks.
LEAST_CONSTRAINING = "least constraining"
