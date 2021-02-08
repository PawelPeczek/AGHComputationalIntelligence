from framework.problems.singleobjective.ackley import Ackley
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from evolutionary_memetic.memetic import MemeticAlgorithm, MemeticLocalSearch

if __name__ == '__main__':
    problem = Ackley(number_of_variables=150)
    max_evaluations = 1000
    mutation = PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20)
    local_search = MemeticLocalSearch(problem, mutation, StoppingByEvaluations(500))
    iterations = 200

    results_memetic = list()
    for i in range(iterations):
        try:
            algorithm = MemeticAlgorithm(
                problem=problem,
                population_size=500,
                offspring_population_size=150,
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )

            algorithm.run()

            results_memetic.append(algorithm.get_result().objectives[0])
            print('{}/{}'.format(i, iterations))
            # print('Algorithm: ' + algorithm.get_name())
            # print('Problem: ' + problem.get_name())
            # print('Solution: ' + str(result.variables[0]))
            # print('Fitness:  ' + str(result.objectives[0]))
            # print('Computing time: ' + str(algorithm.total_computing_time))
        except:
            pass

    import matplotlib.pyplot as plt

    plt.plot(results_memetic)

    result_genetic = []
    for i in range(iterations):
        try:
            algorithm = GeneticAlgorithm(
                problem=problem,
                population_size=500,
                offspring_population_size=150,
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )

            algorithm.run()

            result_genetic.append(algorithm.get_result().objectives[0])
            print('{}/{}'.format(i + iterations, iterations * 2))
            # print('Algorithm: ' + algorithm.get_name())
            # print('Problem: ' + problem.get_name())
            # print('Solution: ' + str(result.variables[0]))
            # print('Fitness:  ' + str(result.objectives[0]))
            # print('Computing time: ' + str(algorithm.total_computing_time))
        except:
            pass

    plt.plot(result_genetic, c='r', label='genetic')
    plt.show()
    import numpy as np

    print(np.mean(results_memetic))
    print(np.mean(result_genetic))
    print(np.mean(results_memetic) < np.mean(result_genetic))
