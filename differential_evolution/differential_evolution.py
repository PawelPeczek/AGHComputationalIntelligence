from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from abc import abstractmethod
from typing import List
import random
import functools


class DEProblem(FloatProblem):
    @abstractmethod
    def affinity(self, solution: FloatSolution):
        pass


class DifferentialEvolution(EvolutionaryAlgorithm[FloatSolution, FloatSolution]):
    def __init__(self,
                 problem: DEProblem,
                 population_size: int,
                 maxiter: int,
                 cr: float,
                 f: float):
        super(EvolutionaryAlgorithm, self).__init__()
        self.population_size = population_size
        self.problem = problem
        self.maxiter = maxiter
        self.dims = problem.number_of_variables
        
        self.cr = cr
        self.f = f
        self.offspring_population_size = population_size

            
    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        return population

    def reproduction(self, population: List[FloatSolution]) -> List[FloatSolution]:
        
        offspring_population = []
        
        for speciman in population:
            a, b, c = random.sample(population, 3)
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
        return self.iter >= self.maxiter

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
    

        