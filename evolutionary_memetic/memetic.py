import copy
import itertools
import random
from abc import ABC
from typing import List

from jmetal.algorithm.singleobjective import LocalSearch
from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm, S, R
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.util.comparator import Comparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion, StoppingByEvaluations


class MemeticLocalSearch(LocalSearch[S, R]):
    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 termination_criterion: StoppingByEvaluations = StoppingByEvaluations(100),
                 comparator: Comparator = store.default_comparator):
        super(MemeticLocalSearch, self).__init__(problem, mutation, termination_criterion, comparator)
        self.solution = None
        self.best_mutated_solution = None

    def set_initial_solution(self, solution: S) -> None:
        self.solution = solution
        self.best_mutated_solution = self.solution

    def create_initial_solutions(self) -> List[S]:
        if self.solution is None:
            raise ValueError("Solution is none")
        self.solutions.append(self.solution)
        return self.solutions

    def step(self) -> None:
        mutated_solution = copy.deepcopy(self.solutions[0])
        mutated_solution: Solution = self.mutation.execute(mutated_solution)
        mutated_solution = self.evaluate([mutated_solution])[0]

        result = self.comparator.compare(mutated_solution, self.best_mutated_solution)
        if result == -1:
            self.best_mutated_solution = mutated_solution
        elif result == 1:
            pass
        else:
            if random.random() < 0.5:
                self.best_mutated_solution = mutated_solution

    def get_result(self) -> R:
        return self.best_mutated_solution

    def get_name(self) -> str:
        return 'Memetic LS'


class MemeticAlgorithm(EvolutionaryAlgorithm[S, R], ABC):
    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 local_search: MemeticLocalSearch,
                 # crossover_rate: int,
                 # mutation_rate: int,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 evaluator: Evaluator = store.default_evaluator):

        super(MemeticAlgorithm, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size)

        self.mutation_op = mutation
        self.crossover_op = crossover
        self.selection_op = selection
        self.local_search = local_search

        # self.crossover_rate = crossover_rate  # To be used
        # self.mutation_rate = mutation_rate  # To be used

        self.evaluator = evaluator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.selection_size = \
            self.offspring_population_size * \
            self.crossover_op.get_number_of_parents() // self.crossover_op.get_number_of_children()

        if self.selection_size < self.crossover_op.get_number_of_children():
            self.selection_size = self.crossover_op.get_number_of_children()

    def selection(self, population: List[S]):
        return [self.selection_op.execute(population) for _ in range(self.selection_size)]

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[R]:

        extended_population = population + offspring_population
        extended_population.sort(key=lambda s: s.objectives[0])

        return extended_population[:self.population_size]

    def reproduction(self, mating_population: List[S]) -> List[R]:
        parents_num_offspring = self.crossover_op.get_number_of_parents()

        parents = [mating_population[i:i + parents_num_offspring] for i in
                   range(0, self.offspring_population_size, parents_num_offspring)]

        offspring = list(itertools.chain.from_iterable([self.crossover_op.execute(par) for par in parents]))

        for i in range(len(offspring)):
            self.local_search.set_initial_solution(offspring[i])
            self.local_search.run()
            offspring[i] = self.local_search.get_result()

        return offspring

    def create_initial_solutions(self) -> List[R]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[S]) -> List[R]:
        return self.evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def get_result(self) -> R:
        self.solutions.sort(key=lambda s: s.objectives[0])
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Memetic algorithm'
