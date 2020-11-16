import threading
import time
from typing import List
import numpy as np

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Mutation
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion


class ClonalSelection(Algorithm[FloatSolution, List[FloatSolution]]):
    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 termination_criterion: TerminationCriterion,
                 selection_size: int,
                 mutation: Mutation,
                 clone_rate: int = 20,
                 random_cells_number: int = 20,
                 evaluator: Evaluator = store.default_evaluator):
        threading.Thread.__init__(self)

        self.solutions: List[FloatSolution] = []
        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.evaluator = evaluator
        self.mutator = mutation

        self.problem = problem
        self.population_size = population_size
        self.selection_size = selection_size
        self.clone_rate = clone_rate
        self.random_cells_number = random_cells_number

        self.observable = store.default_observable

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

    def create_initial_solutions(self) -> List[FloatSolution]:
        """ Creates the initial list of solutions of a metaheuristic. """
        return [self.problem.create_solution() for x in range(self.population_size)]

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        """ Evaluates a solution list. """
        return self.evaluator.evaluate(solution_list, self.problem)

    def init_progress(self) -> None:
        """ Initialize the algorithm. """
        self.evaluations = 0

    def stopping_condition_is_met(self) -> bool:
        """ The stopping condition is met or not. """
        return self.termination_criterion.is_met

    def step(self) -> None:
        """ Performs one iteration/step of the algorithm's loop. """
        affinity_values = [(self.affinity(solution), solution) for solution in self.solutions]
        population_selected = sorted(affinity_values, key=lambda x: x[0], reverse=True)[:self.selection_size]
        clones = []
        for p in population_selected:
            clones += self.clone(p)
        temp_clones = []
        for clone in clones:
            mutated_clone = self.mutator.execute(clone[1])
            temp_clones.append((self.affinity(mutated_clone), mutated_clone))
        clones = temp_clones
        temp_population = affinity_values + clones
        population = sorted(temp_population, key=lambda x: x[0], reverse=True)[:self.population_size]
        population_randoms = [self.problem.create_solution() for x in range(self.random_cells_number)]
        randoms_affinity_values = [(self.affinity(solution), solution) for solution in population_randoms]
        population += randoms_affinity_values
        population = sorted(population, key=lambda x: x[0], reverse=True)[:self.population_size]
        self.solutions = [p[1] for p in population]

    def affinity(self, solution: FloatSolution) -> float:
        direction = [-1 if d == self.problem.MINIMIZE else 1 for d in self.problem.obj_directions]
        return np.multiply(direction, self.problem.evaluate(solution).objectives)

    def clone(self, solution):
        clone_num = int(self.clone_rate / solution[0])
        return [solution] * clone_num

    def update_progress(self) -> None:
        """ Update the progress after each iteration. """
        self.evaluations += 1

        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.solutions
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        """ Get observable data, with the information that will be send to all observers each time. """
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def get_result(self) -> FloatSolution:
        affinity_values = [(self.affinity(solution), solution) for solution in self.solutions]
        result = sorted(affinity_values, key=lambda x: x[0], reverse=True)[0]
        return result[1]

    def get_name(self) -> str:
        return "CLONALG"
