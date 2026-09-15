from dataclasses import replace

from openmind.agent.model.prisoners_dilemma_variant import PrisonersDilemmaVariant

NAME = "prisonersdilemma"
SEPARATOR = "/"
PLAYERS = ("A", "B")
UNSET = None
CERTAIN = 1.0

COOPERATE = "cooperate"
DEFECT = "defect"
CHOICES = (COOPERATE, DEFECT)

#: Axelrod's points for one round: both cooperate, both defect, and a defector against a cooperator.
REWARD = 3
PUNISHMENT = 1
TEMPTATION = 5
SUCKER = 0
POINTS = {
    (COOPERATE, COOPERATE): (REWARD, REWARD),
    (COOPERATE, DEFECT): (SUCKER, TEMPTATION),
    (DEFECT, COOPERATE): (TEMPTATION, SUCKER),
    (DEFECT, DEFECT): (PUNISHMENT, PUNISHMENT),
}

CHOSEN = "chosen"
PLAYED = "played"
ROUND = "round"
SCORE = "score"
TURN = "turn"
PAYOFF = "payoff"
CHOOSE = "choose"
CHOICE = "choice"
ENDING = "ENDING"

STANDARD = PrisonersDilemmaVariant("standard", rounds=10, ending_chance=0.0)
VARIANTS = {
    variant.name: variant
    for variant in (
        STANDARD,
        replace(STANDARD, name="uncertain", rounds=None, ending_chance=0.1),
        replace(STANDARD, name="simultaneous", simultaneous=True),
    )
}
