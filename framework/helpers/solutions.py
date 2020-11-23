from typing import List

from jmetal.core.problem import Problem, S


def initialize_solutions(problem: Problem[S], population_size: int) -> S:
    return [problem.create_solution() for _ in range(population_size)]


def evaluate_solutions(problem: Problem[S], solution_list: List[S]) -> List[S]:
    return [problem.evaluate(s) for s in solution_list]
