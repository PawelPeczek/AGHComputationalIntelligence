from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
import numpy as np


def euclidean_distance(x: Solution, y: Solution) -> float:
    if x.number_of_variables != y.number_of_variables:
        raise ValueError(
            f"Mismatched number of variables in solutions. X: {x},"
            f"Y: {y}"
        )
    x_vect = np.array(x.variables)
    y_vect = np.array(y.variables)
    return np.linalg.norm(x_vect - y_vect)


def solution_comparator(problem: Problem, x: Solution, y: Solution) -> int:
    val1 = problem.evaluate(x).objectives[0]
    val2 = problem.evaluate(y).objectives[0]
    if val1 > val2:
        return 1
    elif val2 > val1:
        return -1
    else:
        return 0
