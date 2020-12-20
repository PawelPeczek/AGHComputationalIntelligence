from math import sqrt, cos

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Griewank(FloatProblem):

    def __init__(self, number_of_variables: int = 10, lower_bound=-5.12, upper_bound=5.12):
        super(Griewank, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [lower_bound for _ in range(number_of_variables)]
        self.upper_bound = [upper_bound for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        sum = 0
        for x in solution.variables:
            sum += x * x
        product = 1
        for i in range(len(solution.variables)):
            product *= cos(solution.variables[i] / sqrt(i + 1))

        solution.objectives[0] = 1 + sum / 4000 - product
        return solution

    def get_name(self) -> str:
        return 'Griewank'