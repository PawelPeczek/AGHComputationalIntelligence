import unittest
from jmetal.algorithm.singleobjective.emas import Emas
from jmetal.operator.neighbours import RandomNeighbours
from jmetal.operator.reproduction import FractionEnergyReproduction
from jmetal.problem import OneMax, Sphere
from jmetal.operator.energy_exchange import FractionEnergyExchange
from jmetal.operator.death import ThresholdDeath
from jmetal.operator.mutation import BitFlipMutation, PolynomialMutation
from jmetal.operator.crossover import SPXCrossover, SBXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations


class EmasTestCase(unittest.TestCase):
    def setUp(self) -> None:
        problem1 = OneMax(number_of_bits=512)

        self.emas1 = Emas(
            problem=problem1,
            initial_population_size=1000,
            initial_inidividual_energy=10,
            reproduction_threshold=20,
            energy_exchange_operator=FractionEnergyExchange(0.5),
            death_operator=ThresholdDeath(threshold=5, neighbours_operator=RandomNeighbours()),
            termination_criterion=StoppingByEvaluations(max_evaluations=100000),
            neighbours_operator=RandomNeighbours(),
            reproduction_operator=FractionEnergyReproduction(0.5, BitFlipMutation(0.5), SPXCrossover(0.5))
        )

        problem2 = Sphere(number_of_variables=10)
        self.emas2 = Emas(
            problem=problem2,
            initial_population_size=1000,
            initial_inidividual_energy=10,
            reproduction_threshold=20,
            energy_exchange_operator=FractionEnergyExchange(0.5),
            death_operator=ThresholdDeath(threshold=5, neighbours_operator=RandomNeighbours()),
            termination_criterion=StoppingByEvaluations(max_evaluations=50000),
            neighbours_operator=RandomNeighbours(),
            reproduction_operator=FractionEnergyReproduction(0.5, PolynomialMutation(0.5), SBXCrossover(0.5))
        )

    def test_lol(self):
        self.emas1.run()
        result = self.emas1.get_result()

        print('Algorithm: ' + self.emas1.get_name())
        print('Solution: ' + str(result.variables[0]))
        print('Fitness:  ' + str(result.objectives[0]))
        print('Computing time: ' + str(self.emas1.total_computing_time))

    def test_lol2(self):
        self.emas2.run()
        result = self.emas2.get_result()

        print('Algorithm: ' + self.emas2.get_name())
        print('Solution: ' + str(result.variables[0]))
        print('Fitness:  ' + str(result.objectives[0]))
        print('Computing time: ' + str(self.emas2.total_computing_time))