from clonal_selection.clonal_selection import ClonalSelection
from clonal_selection.clonal_selection_cognitive import ClonalSelectionCognitive
from clonal_selection.de_jong_1 import DeJong1
from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = DeJong1(-5.12, 5.12, number_of_variables=50)
    max_evaluations = 2000

    clonal_selections = [
        ClonalSelection(
            problem=problem,
            population_size=200,
            selection_size=30,
            random_cells_number=50,
            clone_rate=20,
            mutation=PolynomialMutation(probability=1 / problem.number_of_variables, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        ClonalSelection(
            problem=problem,
            population_size=100,
            selection_size=15,
            random_cells_number=25,
            clone_rate=10,
            mutation=PolynomialMutation(probability=1 / problem.number_of_variables, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )
    ]

    algorithm = ClonalSelectionCognitive(
        clonal_selections=clonal_selections,
        mix_rate=0.4,
        mixes_number=2,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        debug=True
    )

    algorithm.run()

    result = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str([var for var in result.variables]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))

    algorithm.draw_history()
