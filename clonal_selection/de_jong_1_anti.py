import random
import time

from clonal_selection.clonal_selection_anti_elite import ClonalSelectionAntiElite
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from clonal_selection.clonal_selection import ClonalSelection
import matplotlib.pyplot as plt


class DeJong1(FloatProblem):

    def __init__(self, lower_bound,upper_bound, number_of_variables=2):
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
        new_solution = FloatSolution(self.lower_bound, self.upper_bound,
                                     number_of_objectives=self.number_of_objectives)
        for i in range(self.number_of_variables):
            new_solution.variables[i] = random.uniform(self.lower_bound[i], self.upper_bound[i])
        return new_solution

    def get_name(self) -> str:
        return 'DeJong'


if __name__ == '__main__':
    problem = DeJong1(-5.12, 5.12, number_of_variables=50)
    max_evaluations = 1000

    cs_algo = ClonalSelection(
        problem=problem,
        population_size=200,
        selection_size=30,
        random_cells_number=50,
        clone_rate=10,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        debug=True
    )

    cs_algo.run()

    result = cs_algo.get_result()

    print('Algorithm: ' + cs_algo.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str([var for var in result.variables]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(cs_algo.total_computing_time))

    csc_algo = ClonalSelectionAntiElite(
        problem=problem,
        population_size=200,
        selection_size=30,
        random_cells_number=50,
        clone_rate=10,
        mutation_probability=1 / problem.number_of_variables,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        debug=True
    )

    csc_algo.run()

    result = csc_algo.get_result()

    print('Algorithm: ' + csc_algo.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str([var for var in result.variables]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(csc_algo.total_computing_time))

    for o in range(problem.number_of_objectives):
        plt.plot(range(len(csc_algo.history)), [s.objectives[o] for s in csc_algo.history])
    for o in range(problem.number_of_objectives):
        plt.plot(range(len(cs_algo.history)), [s.objectives[o] for s in cs_algo.history])
    legend = [f"objective {i} csc" for i in range(csc_algo.problem.number_of_objectives)] + \
             [f"objective {i} cs" for i in range(cs_algo.problem.number_of_objectives)]
    plt.legend(legend)
    plt.title(f"{problem.get_name()} with {problem.number_of_variables} variables")
    plt.savefig(f"{problem.get_name()}_{problem.number_of_variables}_comparison_history_{time.time()}.jpg")
    plt.show()
