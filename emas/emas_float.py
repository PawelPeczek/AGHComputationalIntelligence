from jmetal.algorithm.singleobjective.emas import Emas
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.operator.death import ThresholdDeath
from jmetal.operator.energy_exchange import FractionEnergyExchange
from jmetal.operator.neighbours import RandomNeighbours
from jmetal.operator.reproduction import FractionEnergyReproduction
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = Sphere(number_of_variables=10)

    algorithm = Emas(
        problem=problem,
        initial_population_size=1000,
        initial_inidividual_energy=10,
        reproduction_threshold=20,
        energy_exchange_operator=FractionEnergyExchange(0.5),
        death_operator=ThresholdDeath(threshold=5, neighbours_operator=RandomNeighbours()),
        termination_criterion=StoppingByEvaluations(max_evaluations=100000),
        neighbours_operator=RandomNeighbours(),
        reproduction_operator=FractionEnergyReproduction(0.5, PolynomialMutation(0.5), SBXCrossover(0.5))
    )

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(result.variables[0]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))
