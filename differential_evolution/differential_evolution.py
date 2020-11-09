import numpy as np
import matplotlib.pyplot as plt
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from abc import abstractmethod
from typing import List
import random
import functools
import math


class DEProblem(FloatProblem):
    @abstractmethod
    def affinity(self, solution: FloatSolution):
        pass


class DifferentialEvolution(EvolutionaryAlgorithm[FloatSolution, FloatSolution]):
    def __init__(self,
                 problem: DEProblem,
                 population_size: int,
                 max_iter: int,
                 cr: float,
                 f: float,
                 is_simulated_annealing: bool = False):
        super(EvolutionaryAlgorithm, self).__init__()
        self.population_size = population_size
        self.problem = problem
        self.max_iter = max_iter
        self.dims = problem.number_of_variables
        
        self.cr = cr
        self.f = f
        self.offspring_population_size = population_size

        ## SIMULATED ANNEALING
        self.is_simulated_annealing = is_simulated_annealing
        self.initial_search_radius = self.calculate_init_search_radius()

            
    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        return population

    def reproduction(self, population: List[FloatSolution]) -> List[FloatSolution]:
        
        offspring_population = []
        
        for speciman in population:
            neighbourhood = population
            
            if self.is_simulated_annealing:
                current_radius = self.initial_search_radius * (1 - self.iter / self.max_iter)
                neighbourhood = list(filter(lambda neighbour: 0 < self.calculate_euclidean_distance(neighbour, speciman) <= current_radius, population))

            if len(neighbourhood) < 3:
                offspring_population.append(speciman)
                continue

            a, b, c = random.sample(neighbourhood, 3)
            dim = random.randint(0, speciman.number_of_variables - 1)
            
            random_weights = [random.random() for _ in range(self.dims)]
            offspring = self.problem.create_solution()
            offspring.variables = speciman.variables.copy()
            
            for i in range(self.dims):
                if random_weights[i] < self.cr or i == dim:
                    offspring.variables[i] = a.variables[i] + self.f * (b.variables[i] - c.variables[i])
            
            offspring_population.append(offspring)
            
        return offspring_population
    
    def replacement(self, population: List[FloatSolution], offspring_population: List[FloatSolution]) -> List[FloatSolution]:
        whole_population = population + offspring_population

        def compare(sol1, sol2) -> int:
            val1 = self.problem.evaluate(sol1).objectives[0]
            val2 = self.problem.evaluate(sol2).objectives[0]
            if val1 > val2:
                return 1
            elif val2 > val1:
                return -1
            else:
                return 0
        
        sorted_whole_population = sorted(whole_population, key=functools.cmp_to_key(compare))
        
        return sorted_whole_population[:len(sorted_whole_population) // 2]
    
    def create_initial_solutions(self) -> List[FloatSolution]:
        """ Creates the initial list of solutions of a metaheuristic. """
        return [self.problem.create_solution() for x in range(self.population_size)]

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        """ Evaluates a solution list. """
        return [self.problem.evaluate(sol) for sol in solution_list]

    def init_progress(self) -> None:
        """ Initialize the algorithm. """
        ###
        self.evaluations = self.population_size
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        ###
        self.iter = 0

    def stopping_condition_is_met(self) -> bool:
        """ The stopping condition is met or not. """
        return self.iter >= self.max_iter

    def update_progress(self) -> None:
        """ Update the progress after each iteration. """
        self.evaluations += self.offspring_population_size
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.iter += 1

    def get_result(self) -> FloatSolution:
        # TODO
        return self.solutions[0]

    def get_name(self) -> str:
        return "DE"

    def calculate_init_search_radius(self) -> float:
        sum_of_squares = 0

        for i in range(self.problem.number_of_variables):
            sum_of_squares += (self.problem.upper_bound[i] - self.problem.lower_bound[i]) ** 2

        return math.sqrt(sum_of_squares)

    def calculate_euclidean_distance(self, x, y) -> float:
        sum_of_squares = 0

        for i in range(x.number_of_variables):
            sum_of_squares += (x.variables[i] - y.variables[i]) ** 2

        return math.sqrt(sum_of_squares)
