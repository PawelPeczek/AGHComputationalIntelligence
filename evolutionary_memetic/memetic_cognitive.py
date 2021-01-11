from typing import List

from evolutionary_memetic.memetic import MemeticAlgorithm, MemeticLocalSearch, MemeticAlgorithm2
from jmetal.config import store
from jmetal.core.algorithm import S, R
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion


class Species:
    def __init__(self,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 local_search: MemeticLocalSearch,
                 termination_criterion: TerminationCriterion):
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.local_search = local_search
        self.termination_criterion = termination_criterion


class MemeticCognitiveAlgorithm(MemeticAlgorithm[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 local_search: MemeticLocalSearch,
                 species1: Species,
                 species2: Species,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 evaluator: Evaluator = store.default_evaluator):
        super(MemeticCognitiveAlgorithm, self).__init__(problem, population_size, offspring_population_size, mutation,
                                                        crossover, selection, local_search, termination_criterion,
                                                        evaluator)
        self.species1: Species = species1
        self.species2: Species = species2
        self.solutions_s1: List[S] = []
        self.solutions_s2: List[S] = []

    def step(self):
        solutions_count = len(self.solutions)
        self.solutions.sort(key=lambda s: s.objectives[0])

        # Najlepsze osobniki ida do gatunku 1
        population_s1 = self.solutions[:(solutions_count // 2)]
        self.solutions_s1 = self._run_memetic(population_s1, self.species1)

        # Najgorsze do gatunku 2
        population_s2 = self.solutions[(solutions_count // 2):]
        self.solutions_s2 = self._run_memetic(population_s2, self.species2)

        self.solutions = self.solutions_s1 + self.solutions_s2

        super(MemeticCognitiveAlgorithm, self).step()

    def _run_memetic(self, initial_population: List[S], species: Species):
        memetic_spec1 = MemeticAlgorithm2(
            problem=self.problem,
            initial_population=initial_population,
            offspring_population_size=self.offspring_population_size // 2,
            mutation=species.mutation,
            crossover=species.crossover,
            selection=species.selection,
            local_search=species.local_search,
            termination_criterion=species.termination_criterion
        )
        memetic_spec1.run()
        return memetic_spec1.solutions
