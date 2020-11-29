import unittest
from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.reproduction import FractionEnergyReproduction
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.problem.singleobjective.knapsack import Knapsack


class FractionEnergyReproductionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        problem = Knapsack(5, 5, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        self.mother = problem.create_solution()
        self.father = problem.create_solution()

        self.mother.energy = 20
        self.father.energy = 40
        self.reproduction_operator = FractionEnergyReproduction(0.5, BitFlipMutation(0.5), SPXCrossover(0.5))

    def test_fraction_energy_reproduction(self):
        mother, father, child = self.reproduction_operator.execute([self.mother, self.father])
        self.assertEqual(mother.energy, 10)
        self.assertEqual(father.energy, 20)
        self.assertEqual(child.energy, 30)
