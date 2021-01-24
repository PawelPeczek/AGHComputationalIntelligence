import time
from framework.problems.singleobjective.ackley import Ackley
from evolutionary_memetic.memetic_cognitive import MemeticCognitiveAlgorithm, Species
import matplotlib.pyplot as plt
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
from evolutionary_memetic.memetic import MemeticAlgorithm, MemeticLocalSearch

if __name__ == '__main__':
    problem = Ackley(number_of_variables=150)
    max_evaluations = 500000
    mutation = PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20)
    local_search = MemeticLocalSearch(problem, mutation, StoppingByEvaluations(500))

    memetic_algo = MemeticCognitiveAlgorithm(
        problem=problem,
        population_size=5000,
        offspring_population_size=1000,
        mutation=mutation,
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(),
        species1=Species(
            mutation=mutation,
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            selection=BinaryTournamentSelection(),
            local_search=local_search,
            termination_criterion=StoppingByEvaluations(max_evaluations=1000)
        ),
        species2=Species(
            mutation=mutation,
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            selection=BinaryTournamentSelection(),
            local_search=local_search,
            termination_criterion=StoppingByEvaluations(max_evaluations=1000)
        ),
        local_search=local_search,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    genetic_algo = GeneticAlgorithm(
        problem=problem,
        population_size=5000,
        offspring_population_size=1000,
        mutation=mutation,
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    memetic_algo.run()
    genetic_algo.run()

    memetic_result = memetic_algo.get_result()
    genetic_result = genetic_algo.get_result()

    print('Algorithm: ' + memetic_algo.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str([var for var in memetic_result.variables]))
    print('Fitness:  ' + str(memetic_result.objectives[0]))
    print('Computing time: ' + str(memetic_algo.total_computing_time))

    print('Algorithm: ' + genetic_algo.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str([var for var in genetic_result.variables]))
    print('Fitness:  ' + str(genetic_result.objectives[0]))
    print('Computing time: ' + str(genetic_algo.total_computing_time))

    for o in range(problem.number_of_objectives):
        plt.plot(range(len(genetic_algo.history)), [s.objectives[o] for s in genetic_algo.history])
    for o in range(problem.number_of_objectives):
        plt.plot(range(len(memetic_algo.history)), [s.objectives[o] for s in memetic_algo.history])


    legend = ["GENETIC", "COGNITIVE-MEMETIC"]

    plt.legend(legend)
    plt.title(f"{problem.get_name()} with {problem.number_of_variables} variables")
    plt.savefig(f"{problem.get_name()}_{problem.number_of_variables}_comparison_history_{time.time()}.jpg")
    plt.show()
