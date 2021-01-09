import time

from clonal_selection.clonal_selection_anti_worse_pro_elite import ClonalSelectionAntiWorseProElite
from clonal_selection.clonal_selection_cognitive import ClonalSelectionCognitive
from clonal_selection.util import get_mean_solution, get_mean_result, get_mean_history
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.operator import PolynomialMutation

from clonal_selection.clonal_selection import ClonalSelection
from framework.problems.singleobjective.griewank import Griewank

import matplotlib.pyplot as plt


def test_griewank():
    problem = Griewank(number_of_variables=50, lower_bound=-100, upper_bound=100)
    max_evaluations = 400
    number_of_tries = 3


    ##################################################################
    results = []
    histories = []
    for i in range(number_of_tries):
        cs_algo = ClonalSelection(
            problem=problem,
            population_size=100,
            selection_size=30,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            debug=True
        )

        cs_algo.run()

        results.append(cs_algo.get_result())
        histories.append([s.objectives[0] for s in cs_algo.history])

    print('Algorithm: ' + cs_algo.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(get_mean_solution(results)))
    print('Fitness:  ' + str(get_mean_result(results)))
    cs_history =  get_mean_history(histories)

    ##################################################################
    results = []
    histories = []
    for _ in range(number_of_tries):
        clonal_selections = [
            ClonalSelection(
                problem=problem,
                population_size=200,
                selection_size=30,
                random_cells_number=100,
                clone_rate=20,
                mutation=PolynomialMutation(probability=1 / problem.number_of_variables, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            ClonalSelection(
                problem=problem,
                population_size=200,
                selection_size=30,
                random_cells_number=50,
                clone_rate=20,
                mutation=PolynomialMutation(probability=1 / problem.number_of_variables * 10, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            ClonalSelection(
                problem=problem,
                population_size=200,
                selection_size=15,
                random_cells_number=25,
                clone_rate=10,
                mutation=PolynomialMutation(probability=1 / problem.number_of_variables, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )
        ]

        csc_algo = ClonalSelectionCognitive(
            clonal_selections=clonal_selections,
            mix_rate=0.4,
            mixes_number=2,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            debug=True
        )

        csc_algo.run()

        results.append(csc_algo.get_result())
        histories.append([s.objectives[0] for s in csc_algo.history])

    print('Algorithm: ' + csc_algo.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(get_mean_solution(results)))
    print('Fitness:  ' + str(get_mean_result(results)))
    csc_history = get_mean_history(histories)

    ##################################################################
    results = []
    histories = []
    for _ in range(number_of_tries):
        csc_algo_anti_pro = ClonalSelectionAntiWorseProElite(
            problem=problem,
            population_size=200,
            selection_size=30,
            random_cells_number=50,
            clone_rate=10,
            mutation_probability=1 / problem.number_of_variables,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            debug=True
        )

        csc_algo_anti_pro.run()

        results.append(csc_algo_anti_pro.get_result())
        histories.append([s.objectives[0] for s in csc_algo_anti_pro.history])

    print('Algorithm: ' + csc_algo_anti_pro.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(get_mean_solution(results)))
    print('Fitness:  ' + str(get_mean_result(results)))
    csc_algo_anti_pro_history = get_mean_history(histories)

    ##################################################################

    for o in range(problem.number_of_objectives):
        plt.plot(range(len(csc_algo_anti_pro_history)), csc_algo_anti_pro_history)
    for o in range(problem.number_of_objectives):
        plt.plot(range(len(csc_history)), csc_history)
    for o in range(problem.number_of_objectives):
        plt.plot(range(len(cs_history)), cs_history)
    legend = [f"objective {i} csc-anti-pro" for i in range(csc_algo_anti_pro.problem.number_of_objectives)] + \
             [f"objective {i} csc-island-model" for i in range(csc_algo.problem.number_of_objectives)] + \
             [f"objective {i} cs" for i in range(cs_algo.problem.number_of_objectives)]
    plt.legend(legend)
    plt.title(f"{problem.get_name()} with {problem.number_of_variables} variables")
    plt.savefig(f"{problem.get_name()}_{problem.number_of_variables}_comparison_history_{time.time()}.jpg")
    plt.show()


if __name__ == '__main__':
    test_griewank()
