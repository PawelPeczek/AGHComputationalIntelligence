from jmetal.core.solution import Solution
from jmetal.core.operator import Reproduction, Mutation, Crossover
from typing import List


class FractionEnergyReproduction(Reproduction[Solution]):
    def __init__(self, fraction: float, mutation_operator: Mutation, crossover_operator: Crossover, **kwargs):
        self.fraction = fraction
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator

    def execute(self, parents: List[Solution]) -> List[Solution]:
        mother, father = parents

        if mother.species_average_fitness > father.species_average_fitness:
            prob = 0.75
            child_species = mother.species_index
        else:
            prob = 0.25
            child_species = father.species_index

        # TODO: Fix Disgusting workaround
        child = self.crossover_operator.__class__(prob).execute(parents)[0]
        child = self.mutation_operator.execute(child)
        child.energy = 0.0

        mother.transfer_energy_to(child, self.fraction*mother.energy)
        father.transfer_energy_to(child, self.fraction*father.energy)

        child.species_index = child_species

        return [mother, father, child]

    def get_name(self) -> str:
        """Fraction energy reproduction operator"""