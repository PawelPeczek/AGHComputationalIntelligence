import unittest
from jmetal.core.solution import BinarySolution, FloatSolution
from jmetal.operator.energy_exchange import FractionEnergyExchange


class StandardEnergyExchangeTestCase(unittest.TestCase):
    def setUp(self):
        self.solution_a = BinarySolution(number_of_objectives=1, number_of_variables=1, initial_energy=20)
        self.solution_a.objectives = [-50.]  # solution_a is worse
        self.solution_b = FloatSolution(lower_bound=[0.1], upper_bound=[0.5], number_of_objectives=1, initial_energy=40)
        self.solution_b.objectives = [-100.]  # solution_b is better

    def test_transfer_energy(self):
        self.solution_a.transfer_energy_to(self.solution_b, 10)
        self.assertEqual(self.solution_b.energy, 50)
        self.assertEqual(self.solution_a.energy, 10)

    def test_transfer_energy_bad_amount(self):
        self.assertRaises(AssertionError, self.solution_a.transfer_energy_to, self.solution_b, 50)

    def test_standard_transfer_energy(self):
        exchange_operator = FractionEnergyExchange(exchange_fraction=0.5)
        exchange_operator.execute([self.solution_a, self.solution_b])

        self.assertEqual(self.solution_a.energy, 10)
        self.assertEqual(self.solution_b.energy, 50)
