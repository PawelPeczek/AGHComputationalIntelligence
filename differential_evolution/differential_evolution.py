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
                 each_species_size: int,
                 max_iter: int,
                 cr: float,
                 f: float,
                 meeting_frequency: int = 20,
                 exchange_rate: float = 0.1,
                 no_species: int = 10,
                 is_simulated_annealing: bool = False,
                 number_of_partners: int = 3):
        super(EvolutionaryAlgorithm, self).__init__()
        self.each_species_size = each_species_size
        self.no_species = no_species
        self.meeting_frequency = meeting_frequency
        self.exchange_rate = exchange_rate
        self.problem = problem
        self.max_iter = max_iter
        self.dims = problem.number_of_variables
        
        self.cr = cr
        self.f = f
        self.offspring_population_size = each_species_size * no_species

        ## SIMULATED ANNEALING
        self.is_simulated_annealing = is_simulated_annealing
        self.initial_search_radius = self.calculate_init_search_radius()

        ## AUGMENTED REPRODUCTION
        self.is_augmented_reproduction = number_of_partners != 3
        self.number_of_partners = number_of_partners

            
    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        return population

    def reproduction(self, population: List[FloatSolution]) -> List[FloatSolution]:

        offspring_population = []

        species_list = self.to_species_list(population)
        
        for speciman in population:
            neighbourhood = species_list[speciman.species_index]
            
            if self.is_simulated_annealing:
                current_radius = self.initial_search_radius * (1 - self.iter / self.max_iter)
                neighbourhood = list(filter(lambda neighbour: 0 < self.calculate_euclidean_distance(neighbour, speciman) <= current_radius, neighbourhood))

            if len(neighbourhood) < self.number_of_partners:
                offspring_population.append(speciman)
                continue
                        
            partners = random.sample(neighbourhood, self.number_of_partners)
            weights = []

            if self.is_augmented_reproduction:
                weights = [math.pow(2, -x) for x in range(1, self.number_of_partners + 1)]
                random.shuffle(weights)

            dim = random.randint(0, speciman.number_of_variables - 1)
            
            random_weights = [random.random() for _ in range(self.dims)]
            offspring = self.problem.create_solution()
            offspring.species_index = speciman.species_index
            offspring.variables = speciman.variables.copy()
            
            for index_dim in range(self.dims):
                if random_weights[index_dim] < self.cr or index_dim == dim:
                    if self.is_augmented_reproduction:
                        value = 0
                        for i, partner in enumerate(partners[1:]):
                            value += weights[i] * partner.variables[index_dim] * (math.pow(-1, random.random() > 0.5))
                        offspring.variables[index_dim] = partners[0].variables[index_dim] + self.f * value
                    else:
                        offspring.variables[index_dim] = partners[0].variables[index_dim] + self.f * (partners[1].variables[index_dim] - partners[2].variables[index_dim])
            
            offspring_population.append(offspring)
            
        return offspring_population
    
    def replacement(self, population: List[FloatSolution], offspring_population: List[FloatSolution]) -> List[FloatSolution]:        
        offspring_species_list = self.to_species_list(offspring_population)
        species_list = self.to_species_list(population)

        whole_population = list()
        for i in range(self.no_species):
            whole_species = offspring_species_list[i] + species_list[i]
            sorted_whole_species = sorted(whole_species, key=functools.cmp_to_key(self.compare))
            whole_population += sorted_whole_species[:len(sorted_whole_species) // 2]

        ## EXCHANGE
        if self.iter % self.meeting_frequency == 0:
            species_to_exchange = list()
            for i in range(self.no_species):
                species_to_exchange.append(list())
    
            no_species_to_exchange = int(self.exchange_rate * self.each_species_size)
            species_list = self.to_species_list(whole_population)

            for i in range(len(species_list)):
                next_index = (i + 1) % len(species_list)
                for speciman in species_list[i][0:no_species_to_exchange]:
                    speciman_copy = self.problem.create_solution()
                    speciman_copy.species_index = next_index
                    speciman_copy.variables = speciman.variables.copy()
                    species_to_exchange[next_index].append(speciman_copy)

            whole_population.clear()
            for i in range(len(species_list)):
                whole_species = species_list[i] + species_to_exchange[i]
                sorted_whole_species = sorted(whole_species, key=functools.cmp_to_key(self.compare))
                whole_population += sorted_whole_species[:self.each_species_size]
        
        return whole_population
    
    def create_initial_solutions(self) -> List[FloatSolution]:
        """ Creates the initial list of solutions of a metaheuristic. """
        population = [self.problem.create_solution() for x in range(self.each_species_size * self.no_species)]

        for i in range(len(population)):
            population[i].species_index = i % self.no_species
        
        return population

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        """ Evaluates a solution list. """
        return [self.problem.evaluate(sol) for sol in solution_list]

    def init_progress(self) -> None:
        """ Initialize the algorithm. """
        ###
        self.evaluations = self.each_species_size
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
        return sorted(self.solutions, key=functools.cmp_to_key(self.compare))[0]

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

    def to_species_list(self, population) -> List[List[FloatSolution]]:
        species_list = list()
        
        for species in range(self.no_species):
            species_list.append(list())

        for speciman in population:
            species_list[speciman.species_index].append(speciman)

        return species_list

    def compare(self, sol1, sol2) -> int:
        val1 = self.problem.evaluate(sol1).objectives[0]
        val2 = self.problem.evaluate(sol2).objectives[0]
        if val1 > val2:
            return 1
        elif val2 > val1:
            return -1
        else:
            return 0
        