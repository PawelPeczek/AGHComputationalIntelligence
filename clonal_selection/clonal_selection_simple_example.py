import random

from jmetal.core.solution import FloatSolution
from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from clonal_selection import ClonalProblem, ClonalSelection


class QuadraticFunction(ClonalProblem):

    def __init__(self, upper_bound, lower_bound):
        super(QuadraticFunction, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.upper_bound = [upper_bound] * self.number_of_variables
        self.lower_bound = [lower_bound] * self.number_of_variables

        self.obj_directions = [self.MINIMIZE]

        self.obj_labels = ['y1']

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        y1 = x1 ** 2
        solution.objectives[0] = y1
        return solution

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.lower_bound, self.lower_bound,
                                     number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = random.uniform(self.lower_bound[0], self.upper_bound[0])

        return new_solution

    def affinity(self, solution: FloatSolution) -> float:
        return - solution.variables[0] ** 2

    def get_name(self) -> str:
        return 'Q-F'


if __name__ == '__main__':
    problem = QuadraticFunction(-5, 5)
    max_evaluations = 1000

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
    print('Solution: ' + str(result.variables[0]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))
