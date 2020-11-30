import unittest
from jmetal.operator.death import ThresholdDeath, NoIndividualsAliveException
from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.neighbours import RandomNeighbours


class ThresholdDeathTestCase(unittest.TestCase):
    def setUp(self):
        self.solution_a = BinarySolution(number_of_objectives=1, number_of_variables=1, initial_energy=20)
        self.solution_b = FloatSolution(lower_bound=[0.1], upper_bound=[0.5], number_of_objectives=1, initial_energy=40)
        self.solution_c = BinarySolution(number_of_objectives=1, number_of_variables=1, initial_energy=30)
        self.solutions = [self.solution_a, self.solution_b, self.solution_c]

    def test_proper_execution(self):
        death_operator = ThresholdDeath(30, neighbours_operator=RandomNeighbours(seed=1))
        total_energy_before = sum([x.energy for x in self.solutions])

        after_death_solutions = death_operator.execute(self.solutions)

        total_energy_after = sum([x.energy for x in after_death_solutions])

        self.assertEqual(len(after_death_solutions), 2)
        self.assertTrue(self.solution_b in after_death_solutions)
        self.assertTrue(self.solution_c in after_death_solutions)
        self.assertEqual(total_energy_after, total_energy_before)
        self.assertTrue((self.solution_b.energy == 60 and self.solution_c.energy == 30) or
                        (self.solution_b.energy == 40 and self.solution_c.energy == 50))

    def test_proper_execution_2(self):
        death_operator = ThresholdDeath(10, neighbours_operator=RandomNeighbours(seed=1))
        total_energy_before = sum([x.energy for x in self.solutions])
        after_death_solutions = death_operator.execute(self.solutions)
        total_energy_after = sum([x.energy for x in after_death_solutions])

        self.assertEqual(len(after_death_solutions), 3)
        self.assertTrue(self.solution_a in after_death_solutions)
        self.assertTrue(self.solution_b in after_death_solutions)
        self.assertTrue(self.solution_c in after_death_solutions)
        self.assertEqual(total_energy_after, total_energy_before)
        self.assertEqual(self.solution_a.energy, 20)
        self.assertEqual(self.solution_b.energy, 40)
        self.assertEqual(self.solution_c.energy, 30)

    def test_raises_no_alive_individuals(self):
        death_operator = ThresholdDeath(50, neighbours_operator=RandomNeighbours(seed=1))
        self.assertRaises(NoIndividualsAliveException, death_operator.execute, self.solutions)




