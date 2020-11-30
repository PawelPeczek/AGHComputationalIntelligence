import unittest

from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.neighbours import RandomNeighbours
import random


class RandomNeighboursTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.solution_a = BinarySolution(number_of_objectives=1, number_of_variables=1, initial_energy=20)
        self.solution_b = FloatSolution(lower_bound=[0.1], upper_bound=[0.5], number_of_objectives=1, initial_energy=40)
        self.solution_c = BinarySolution(number_of_objectives=1, number_of_variables=2, initial_energy=30)
        self.solution_d = BinarySolution(number_of_objectives=1, number_of_variables=2, initial_energy=30)
        self.neighbours_operator = RandomNeighbours(seed=1)

    def test_random_neighbours(self):
        individual = self.solution_a
        rest = [self.solution_b, self.solution_c, self.solution_d]

        neigh = self.neighbours_operator.execute((individual, rest))
        self.assertEqual(neigh, self.solution_b)

    def test_no_neighbours_left(self):
        individual = self.solution_a
        neigh = self.neighbours_operator.execute((individual, []))
        self.assertIsNone(neigh)
