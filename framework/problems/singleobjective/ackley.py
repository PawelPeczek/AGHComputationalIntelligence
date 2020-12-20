from math import sqrt, cos, pi, exp

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Ackley(FloatProblem):

    def __init__(self, number_of_variables: int = 10, lower_bound: float = -32.768, upper_bound: float = 32.768):
        super(Ackley, self).__init__()
        assert lower_bound < upper_bound
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
        dim = solution.number_of_variables

        part_1 = -0.2 * sqrt(1/dim * sum([x**2 for x in solution.variables]))
        part_2 = 1/dim * sum([cos(2 * pi * x) for x in solution.variables])
        result = -20 * exp(part_1) - exp(part_2) + exp(1) + 20

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Ackley'