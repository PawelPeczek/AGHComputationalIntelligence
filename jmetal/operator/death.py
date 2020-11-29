from jmetal.core.operator import Death, Neighbours
from jmetal.core.solution import Solution
from jmetal.operator.neighbours import RandomNeighbours
from typing import List
import random


class NoIndividualsAliveException(Exception):
    def __init__(self):
        pass


class ThresholdDeath(Death[Solution]):
    def __init__(self, threshold: float, neighbours_operator: Neighbours = RandomNeighbours()):
        self.threshold = threshold
        self.neighbours_operator = neighbours_operator

    def execute(self, individuals: List[Solution]) -> List[Solution]:
        alive_solutions = list(filter(lambda individual: individual.energy >= self.threshold, individuals))

        if not alive_solutions:
            raise NoIndividualsAliveException()

        to_kill_solutions = list(filter(lambda individual: individual.energy < self.threshold, individuals))

        transfered = []

        for to_kill in to_kill_solutions:
            alive_neighbour = self.neighbours_operator.execute((to_kill, alive_solutions))
            to_kill.transfer_energy_to(alive_neighbour, to_kill.energy)
            alive_solutions.remove(alive_neighbour)
            transfered.append(alive_neighbour)

        return transfered + alive_solutions

    def get_name(self) -> str:
        "Treshhold Death Operator"
