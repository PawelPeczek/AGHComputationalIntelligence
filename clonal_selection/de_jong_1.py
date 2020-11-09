import random

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from clonal_selection import ClonalSelection


class DeJong1(FloatProblem):

    def __init__(self, upper_bound, lower_bound, number_of_variables=2):
        super(DeJong1, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.upper_bound = [upper_bound] * self.number_of_variables
        self.lower_bound = [lower_bound] * self.number_of_variables

        self.obj_directions = [self.MINIMIZE] * self.number_of_objectives

        self.obj_labels = ['y']

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        s = 0
        for x in solution.variables:
            s += x ** 2
        solution.objectives[0] = s
        return solution

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.lower_bound, self.lower_bound,
                                     number_of_objectives=self.number_of_objectives)
        for i in range(self.number_of_variables):
            new_solution.variables[i] = random.uniform(self.lower_bound[i], self.upper_bound[i])
        return new_solution

    def get_name(self) -> str:
        return 'DeJong'


if __name__ == '__main__':
    problem = DeJong1(-5.12, 5.12)
    max_evaluations = 25000

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
    print('Solution: ' + str(result.variables[0]) + " " + str(result.variables[1]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))
