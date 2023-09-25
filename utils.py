from enum import Enum


class WeightingMethod(Enum):
    AVG = "avg"
    CONST = "const"


class Method(Enum):
    EPS_GREEDY = "eps_greedy"
    UCB = "ucb"
    GRADIENT = "gradient"
