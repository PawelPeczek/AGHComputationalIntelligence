import time

from clonal_selection.clonal_selection_anti_worse_pro_elite import ClonalSelectionAntiWorseProElite
from clonal_selection.clonal_selection_cognitive import ClonalSelectionCognitive
from framework.problems.singleobjective.schwefel import Schwefel
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.operator import PolynomialMutation, SimpleRandomMutation

from clonal_selection.clonal_selection import ClonalSelection

import matplotlib.pyplot as plt


def test_schwefel():
    problem = Schwefel(number_of_variables=50, lower_bound=-500, upper_bound=500)
    max_evaluations = 1000

    cs_algo = ClonalSelection(
        problem=problem,
        population_size=100,
        selection_size=30,
        mutation=SimpleRandomMutation(probability=1.0 / problem.number_of_variables * 5),
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

    ##################################################################

    max_evaluations = 1000

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
            random_cells_number=100,
            clone_rate=20,
            mutation=PolynomialMutation(probability=1 / problem.number_of_variables * 10, distribution_index=2),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        ClonalSelection(
            problem=problem,
            population_size=200,
            selection_size=15,
            random_cells_number=100,
            clone_rate=10,
            mutation=PolynomialMutation(probability=1 / problem.number_of_variables, distribution_index=200),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        ClonalSelection(
            problem=problem,
            population_size=200,
            selection_size=15,
            random_cells_number=50,
            clone_rate=10,
            mutation=SimpleRandomMutation(probability=1 / problem.number_of_variables * 5),
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

    result = csc_algo.get_result()

    print('Algorithm: ' + csc_algo.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str([var for var in result.variables]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(csc_algo.total_computing_time))

    ##################################################################

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

    result = csc_algo_anti_pro.get_result()

    print('Algorithm: ' + csc_algo_anti_pro.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str([var for var in result.variables]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(csc_algo_anti_pro.total_computing_time))

    ##################################################################

    for o in range(problem.number_of_objectives):
        plt.plot(range(len(csc_algo_anti_pro.history)), [s.objectives[o] for s in csc_algo_anti_pro.history])
    for o in range(problem.number_of_objectives):
        plt.plot(range(len(csc_algo.history)), [s.objectives[o] for s in csc_algo.history])
    for o in range(problem.number_of_objectives):
        plt.plot(range(len(cs_algo.history)), [s.objectives[o] for s in cs_algo.history])
    legend = [f"objective {i} csc-anti-pro" for i in range(csc_algo_anti_pro.problem.number_of_objectives)] + \
             [f"objective {i} cs" for i in range(cs_algo.problem.number_of_objectives)] + \
             [f"objective {i} csc-island-model" for i in range(csc_algo.problem.number_of_objectives)]
    plt.legend(legend)
    plt.title(f"{problem.get_name()} with {problem.number_of_variables} variables")
    plt.savefig(f"{problem.get_name()}_{problem.number_of_variables}_comparison_history_{time.time()}.jpg")
    plt.show()


if __name__ == '__main__':
    test_schwefel()
