import threading
from jmetal.core.algorithm import Algorithm
from typing import List
from jmetal.core.operator import EnergyExchange, Death, Neighbours, Operator, Reproduction
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.util.generator import Generator, RandomGenerator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.evaluator import Evaluator, SequentialEvaluator
from copy import copy
import itertools
import time
import numpy as np


class Emas(Algorithm[Solution, Solution], threading.Thread):
    """Emas algorithm implementation"""

    def __init__(self,
                 problem: Problem,
                 initial_population_size: int,
                 initial_inidividual_energy: float,
                 reproduction_threshold: float,
                 energy_exchange_operator: EnergyExchange,
                 death_operator: Death,
                 termination_criterion: TerminationCriterion,
                 neighbours_operator: Neighbours,
                 reproduction_operator: Reproduction,
                 population_generator: Generator = RandomGenerator(),
                 population_evaluator: Evaluator = SequentialEvaluator(),
                 no_species=1,
                 species_size=0):

        assert species_size * no_species == initial_population_size

        super(Emas, self).__init__(no_species, species_size)
        self.reproduction_operator = reproduction_operator
        self.reproduction_threshold = reproduction_threshold
        self.initial_inidividual_energy = initial_inidividual_energy
        self.neighbours_operator = neighbours_operator
        self.problem = problem
        self.initial_population_size = initial_population_size
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.death_operator = death_operator
        self.energy_exchange_operator = energy_exchange_operator

        self.species_fitness = {i: 0.0 for i in range(no_species)}

        self.evaluations_history = []
        self.fitness_history = []

    def create_initial_solutions(self) -> List[Solution]:
        initial_solutions = [self.population_generator.new(self.problem)
                             for _ in range(self.initial_population_size)]

        for solution in initial_solutions:
            solution.energy = float(self.initial_inidividual_energy)

        self.determine_species(initial_solutions)
        return initial_solutions

    def evaluate(self, solution_list: List[Solution]) -> List[Solution]:
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def init_progress(self) -> None:
        self.evaluations = self.initial_population_size
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def _update_species_fitness(self, solution_list) -> None:
        separated_solutions = self.to_species_list(solution_list)
        for species in range(self.no_species):
            self.species_fitness[species] = np.average(
                list(map(lambda x: x.objectives[0], separated_solutions[species])))
            for x in separated_solutions[species]:
                x.species_average_fitness = self.species_fitness[species]

    def _neighbours_operation(self, solution_list: List[Solution], operator: Operator[Solution, Solution]) -> List[
        Solution]:
        solution_list = [copy(x) for x in solution_list]

        all_done = []

        while solution_list:
            neigh_a = solution_list.pop()
            neigh_b = self.neighbours_operator.execute((neigh_a, solution_list))
            if neigh_b is None:
                all_done += [neigh_a]
                break

            solution_list.remove(neigh_b)
            all_done += operator.execute([neigh_a, neigh_b])

        return all_done

    def exchange_energy(self, solution_list: List[Solution]) -> List[Solution]:
        copied_solution_list = [copy(x) for x in solution_list]
        return self._neighbours_operation(copied_solution_list, self.energy_exchange_operator)

    def reproduce(self, solution_list: List[Solution]) -> List[Solution]:
        fit_for_reproduction = list(filter(lambda x: x.energy >= self.reproduction_threshold, solution_list))
        not_fit_for_reproduction = list(filter(lambda x: x.energy < self.reproduction_threshold, solution_list))

        copied_fit = [copy(x) for x in fit_for_reproduction]
        copied_not_fit = [copy(x) for x in not_fit_for_reproduction]

        return self._neighbours_operation(copied_fit, self.reproduction_operator) + copied_not_fit

    def kill(self, solution_list: List[Solution]) -> List[Solution]:
        copied_solutions = [copy(x) for x in solution_list]
        return self.death_operator.execute(copied_solutions)

    def step(self) -> None:
        x = self.solutions
        x = self.evaluate(x)
        self._update_species_fitness(x)
        # print(self.species_fitness)
        # print(list(map(lambda species: len(species), self.to_species_list(x))))
        x = self.exchange_energy(x)
        x = self.kill(x)
        x = self.reproduce(x)
        x = sorted(x, key=lambda sol: sol.objectives[0])
        self.solutions = x

        self.fitness_history.append(self.get_result().objectives[0])
        self.evaluations_history.append(self.evaluations)
        # print(self.evaluations)

    def update_progress(self) -> None:
        self.evaluations += len(self.solutions)

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'SPECIES': self.get_result().species_index,
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def get_result(self) -> Solution:
        return self.solutions[0]

    def get_name(self) -> str:
        return "Emas algorithm"
