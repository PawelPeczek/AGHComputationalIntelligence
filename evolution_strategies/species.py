from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy, S
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.constraint_handling import overall_constraint_violation_degree

from jmetal.util.evaluator import Evaluator, SequentialEvaluator
from jmetal.util.generator import Generator, RandomGenerator
from jmetal.core.problem import Problem
from typing import List
from copy import copy
from collections import namedtuple
import time

from jmetal.core.solution import FloatSolution
from jmetal.core.algorithm import Algorithm

from evolution_strategies.cognitive_mutation import CognitivePolynomialMutation

StrategyParams = namedtuple('StrategyParams', ['mu', 'lambda_', 'elitist', 'look_at_others_probability'])


class SocioCognitiveEvolutionStrategy(Algorithm[FloatSolution, FloatSolution]):
    def __init__(self,
                 problem: Problem,
                 strategies_params: List[StrategyParams],
                 termination_criterion: TerminationCriterion,
                 population_generator: Generator = RandomGenerator(),
                 population_evaluator: Evaluator = SequentialEvaluator()):
        super(SocioCognitiveEvolutionStrategy, self).__init__()

        self.strategies = [EvolutionStrategy(problem=problem,
        mu=params.mu,
        lambda_=params.lambda_,
        elitist=params.elitist,
        mutation=CognitivePolynomialMutation(probability=1.0 / problem.number_of_variables, look_at_others_probability=params.look_at_others_probability),
        termination_criterion=termination_criterion) for params in strategies_params]

        self.problem = problem

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.population_generator = population_generator
        self.population_evaluator = population_evaluator

        self.history = []

    def create_initial_solutions(self):
        """ Creates the initial list of solutions of a metaheuristic. """
        all_solutions = []
        for es in self.strategies:
            initial_solutions = es.create_initial_solutions()
            es.solutions = initial_solutions
            all_solutions += initial_solutions
            
        return all_solutions
    
    def evaluate(self, solution_list):
        """ Evaluates a solution list. """
        all_solutions = []
        for es in self.strategies:
            evaluated_solutions = es.evaluate(es.solutions)
            es.solutions = evaluated_solutions
            all_solutions += evaluated_solutions
            
        return all_solutions
    
    def init_progress(self) -> None:
        for es in self.strategies:
            es.init_progress()
    
    def step(self):
        all_solutions = []
        for es in self.strategies:
            es.step()
            all_solutions += es.solutions
            
        self.solutions = all_solutions
        self.solutions.sort(key=lambda x: x.objectives[0], reverse=False)
        ## TODO: update mutation population
        self.update_history()
    
    def update_history(self):
        best_fitness = self.solutions[0].objectives[0]
        self.history.append(best_fitness)

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def init_progress(self) -> None:
        for es in self.strategies:
            self.evaluations += es.population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def update_progress(self) -> None:
        for es in self.strategies:
            self.evaluations += es.offspring_population_size

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def select_final_population(self, population: List[S], mu: int) -> List[S]:
        population_pool = copy(population)
        population_pool.sort(key=lambda s: (
            overall_constraint_violation_degree(s), s.objectives[0]))

        new_population = []
        for i in range(mu):
            new_population.append(population_pool[i])

        return new_population

    def stopping_condition_is_met(self) -> bool:
        """ The stopping condition is met or not. """
        return self.termination_criterion.is_met
    
    def get_history(self):
        return self.history

    def get_result(self):
        return self.solutions[0]

    def get_name(self) -> str:
        return 'SocioCognitiveEvolutionStrategy'
