from math import sqrt, sin

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class SchafferF7(FloatProblem):

    def __init__(self, number_of_variables: int = 10, lower_bound: float = -100.0, upper_bound: float = 100.0):
        super(SchafferF7, self).__init__()
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

        part_1 = 1 / (dim - 1)

        result = 0
        for i in range(dim - 1):
            part_2 = sqrt(solution.variables[i] ** 2 + solution.variables[i + 1] ** 2)
            part_3 = sin(50.0 * part_2 ** 0.2) + 1

            result += (part_1 * part_2 * part_3) ** 2

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'SchafferF7'