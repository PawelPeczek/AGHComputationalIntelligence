from math import sqrt, sin

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class SchafferF6(FloatProblem):

    def __init__(self, lower_bound: float = -100.0, upper_bound: float = 100.0):
        super(SchafferF6, self).__init__()
        assert lower_bound < upper_bound
        number_of_variables = 2
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
        x = solution.variables[0]
        y = solution.variables[1]

        part_1 = sin(sqrt(x ** 2 + y ** 2)) ** 2 - 0.5
        part_2 = (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

        result = 0.5 + part_1 / part_2

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'SchafferF6'