import random

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from clonal_selection import ClonalSelection


class DeJong5(FloatProblem):

    def __init__(self, upper_bound, lower_bound):
        super(DeJong5, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = 2
        self.number_of_constraints = 0

        self.upper_bound = [upper_bound] * self.number_of_variables
        self.lower_bound = [lower_bound] * self.number_of_variables

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives

        self.obj_labels = ['y']

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[0]
        a = np.array([-32, -16, 0, 16, 32])
        a1 = np.tile(a, 5)
        a2 = np.repeat(a, 5)
        A = np.array([a1, a2])
        s = 0
        for i in range(25):
            s += 1 / (i + (x1 - A[0][i]) ** 6 + (x2 - A[1][i]) ** 6)
        y1 = (0.002 + s) ** (-1)
        solution.objectives[0] = y1
        return solution

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.lower_bound, self.lower_bound,
                                     number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = random.uniform(self.lower_bound[0], self.upper_bound[0])
        new_solution.variables[1] = random.uniform(self.lower_bound[1], self.upper_bound[1])

        return new_solution

    def get_name(self) -> str:
        return 'DeJong'


if __name__ == '__main__':
    problem = DeJong5(-5, 5)
    max_evaluations = 500

    algorithm = ClonalSelection(
        problem=problem,
        population_size=100,
        selection_size=30,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.run()

    result = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(result.variables[0]) + str(result.variables[0]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))
