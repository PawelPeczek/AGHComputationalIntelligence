from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from abc import abstractmethod
from typing import List
import random
import functools
import math

from framework.helpers.metrics import euclidean_distance, solution_comparator
from framework.helpers.termination_criterion import EnrichedStoppingByEvaluations
from framework.helpers.solutions import initialize_solutions, evaluate_solutions


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
                 is_simulated_annealing: bool = False,
                 number_of_partners: int = 3):
        super(EvolutionaryAlgorithm, self).__init__()
        self.population_size = population_size
        self.problem = problem
        self.dims = problem.number_of_variables
        
        self.cr = cr
        self.f = f
        self.offspring_population_size = population_size

        ## SIMULATED ANNEALING
        self.is_simulated_annealing = is_simulated_annealing
        self.initial_search_radius = self.calculate_init_search_radius()

        ## AUGMENTED REPRODUCTION
        self.is_augmented_reproduction = number_of_partners != 3
        self.number_of_partners = number_of_partners
        self.__stop_criterion = EnrichedStoppingByEvaluations(max_iter)
        self.observable.register(self.__stop_criterion)

    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        return population

    def reproduction(self, population: List[FloatSolution]) -> List[FloatSolution]:
        
        offspring_population = []
        
        for speciman in population:
            neighbourhood = population
            
            if self.is_simulated_annealing:
                current_radius = self.initial_search_radius * (1 - self.__stop_criterion.progress)
                neighbourhood = list(filter(lambda neighbour: 0 < euclidean_distance(neighbour, speciman) <= current_radius, population))

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
        whole_population = population + offspring_population
        sorted_whole_population = sorted(whole_population, key=functools.cmp_to_key(solution_comparator))
        return sorted_whole_population[:len(sorted_whole_population) // 2]
    
    def create_initial_solutions(self) -> List[FloatSolution]:
        """ Creates the initial list of solutions of a metaheuristic. """
        return initialize_solutions(
            problem=self.problem,
            population_size=self.population_size
        )

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        """ Evaluates a solution list. """
        return evaluate_solutions(
            problem=self.problem,
            solution_list=solution_list
        )

    def init_progress(self) -> None:
        """ Initialize the algorithm. """
        ###
        self.evaluations = self.population_size
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        ###
        self.__stop_criterion.reset()

    def stopping_condition_is_met(self) -> bool:
        """ The stopping condition is met or not. """
        return self.__stop_criterion.is_met

    def update_progress(self) -> None:
        """ Update the progress after each iteration. """
        self.evaluations += self.offspring_population_size
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

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
