import random
import threading
from typing import List, Tuple

import numpy as np
from sortedcontainers import SortedList

from clonal_selection.clonal_selection import ClonalSelection
from jmetal.config import store
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion, StoppingByEvaluations


class ClonalSelectionAntiWorseProElite(ClonalSelection):
    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 selection_size: int,
                 mutation_probability: float,
                 clone_rate: int = 20,
                 random_cells_number: int = 20,
                 termination_criterion: TerminationCriterion = StoppingByEvaluations(max_evaluations=500),
                 evaluator: Evaluator = store.default_evaluator,
                 pull_probability: float = 0.33,
                 push_probability: float = 0.33,
                 random_probability: float = 0.33,
                 debug: bool = False):
        threading.Thread.__init__(self)

        self.solutions: SortedList[Tuple[FloatSolution, float]] = SortedList([], key=lambda x: -x[1])
        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.evaluator = evaluator

        self.problem = problem
        self.population_size = population_size
        self.selection_size = selection_size
        self.clone_rate = clone_rate
        self.random_cells_number = random_cells_number

        self.observable = store.default_observable

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.history: List[FloatSolution] = []

        self.mutation_probability = mutation_probability
        self.pull_probability = pull_probability
        self.push_probability = push_probability
        self.random_probability = random_probability

        self.debug = debug

    def step(self) -> None:
        """ Performs one iteration/step of the algorithm's loop. """
        elite_population_selected = SortedList(self.solutions[:self.selection_size], key=lambda x: -x[1])
        worse_population_selected = SortedList(self.solutions[:-self.selection_size], key=lambda x: x[1])
        max_affinity = elite_population_selected[0][1]
        clones = []
        for p in elite_population_selected:
            clone = self.clone(p, max_affinity)
            clones += clone
        temp_clones = SortedList([], key=lambda x: -x[1])
        mean_best_specie = self.get_mean_specie(elite_population_selected)
        mean_worse_specie = self.get_mean_specie(worse_population_selected)
        for clone in clones:
            mutated_clone = self.mutate(clone, mean_best_specie, mean_worse_specie)
            temp_clones.add((mutated_clone, self.affinity(mutated_clone)))
        clones = temp_clones
        self.solutions.update(clones)
        population_randoms = self.generate_random()
        self.solutions.update(population_randoms)
        self.solutions = SortedList(self.solutions[:self.population_size], key=lambda x: -x[1])
        self.update_history()

    @staticmethod
    def get_mean_specie(selected_population):
        return np.mean([x[0].variables for x in selected_population], axis=0)

    def mutate(self, specie, mean_elite_specie, mean_worse_specie):
        for i in range(self.problem.number_of_variables):
            rand = random.random()
            if rand <= self.mutation_probability:
                # odpychanie
                if random.random() < self.push_probability:
                    if mean_worse_specie[i] > specie.variables[i]:
                        specie.variables[i] -= (specie.variables[i] - specie.lower_bound[i]) * random.random()
                    else:
                        specie.variables[i] += (specie.upper_bound[i] - specie.variables[i]) * random.random()

                # przyciąganie
                if random.random() < self.pull_probability:
                    if mean_elite_specie[i] > specie.variables[i]:
                        specie.variables[i] += (specie.upper_bound[i] - specie.variables[i]) * random.random()
                    else:
                        specie.variables[i] -= (specie.variables[i] - specie.lower_bound[i]) * random.random()
                # losowa mutacja
                if random.random() < self.random_probability:
                    specie.variables[i] = specie.lower_bound[i] + \
                                            (specie.upper_bound[i] - specie.lower_bound[i]) * random.random()

        return specie

    def get_name(self) -> str:
        return "CLONAL-ANTI-PRO"
