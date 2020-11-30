from jmetal.core.operator import EnergyExchange
from jmetal.core.solution import Solution
from typing import List
from jmetal.core.solution import BinarySolution
from jmetal.util.comparator import DominanceComparator


class FractionEnergyExchange(EnergyExchange[Solution]):
    def __init__(self, exchange_fraction: float, ):
        self.exchange_factor = exchange_fraction

    def execute(self, individuals: List[Solution]) -> List[Solution]:
        assert len(individuals) == 2

        sol_a, sol_b = individuals
        better_sol, worse_sol = sol_a, sol_b

        if DominanceComparator().compare(sol_a, sol_b):
            better_sol, worse_sol = sol_b, sol_a

        worse_sol.transfer_energy_to(better_sol, self.exchange_factor*worse_sol.energy)

        return [sol_a, sol_b]

    def get_name(self) -> str:
        "Fraction Energy Exchange Operator"
