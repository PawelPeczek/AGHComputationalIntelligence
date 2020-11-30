import threading
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedList

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

        self.solutions: SortedList[Tuple[FloatSolution, float]] = SortedList([], key=lambda x: -x[1])
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
        self.history: List[FloatSolution] = []

    def update_history(self):
        best_solution = self.get_result()
        self.history.append(best_solution)

    def create_initial_solutions(self):  # -> SortedList[Tuple[FloatSolution, float]]:
        """ Creates the initial list of solutions of a metaheuristic. """
        solutions = []
        for _ in range(self.population_size):
            solution = self.problem.create_solution()
            solutions.append((solution, self.affinity(solution)))

        return SortedList(solutions, key=lambda x: -x[1], )

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        """ Evaluates a solution list. """
        self.evaluator.evaluate([s[0] for s in solution_list], self.problem)
        return solution_list

    def init_progress(self) -> None:
        """ Initialize the algorithm. """
        self.evaluations = 0

    def stopping_condition_is_met(self) -> bool:
        """ The stopping condition is met or not. """
        return self.termination_criterion.is_met

    def step(self) -> None:
        """ Performs one iteration/step of the algorithm's loop. """
        population_selected = SortedList(self.solutions[:self.selection_size], key=lambda x: -x[1])
        max_affinity = population_selected[0][1]
        clones = []
        for p in population_selected:
            clone = self.clone(p, max_affinity)
            clones += clone
        temp_clones = SortedList([], key=lambda x: -x[1])
        for clone in clones:
            mutated_clone = self.mutator.execute(clone)
            temp_clones.add((mutated_clone, self.affinity(mutated_clone)))
        clones = temp_clones
        self.solutions.update(clones)
        population_randoms = self.generate_random()
        self.solutions.update(population_randoms)
        self.solutions = SortedList(self.solutions[:self.population_size], key=lambda x: -x[1])
        self.update_history()

    def generate_random(self):  # -> SortedList[Tuple[FloatSolution, float]]:
        randoms = []
        for _ in range(self.random_cells_number):
            solution = self.problem.create_solution()
            affinity = self.affinity(solution)
            randoms.append((solution, affinity))
        return SortedList(randoms, key=lambda x: -x[1])

    def affinity(self, solution: FloatSolution) -> float:
        direction = [-1 if d == self.problem.MINIMIZE else 1 for d in self.problem.obj_directions]
        return np.multiply(direction, self.problem.evaluate(solution).objectives)

    def clone(self, solution, max_affinity):
        clone_num = int(self.clone_rate * (abs(max_affinity/solution[1])))
        clones = []
        for i in range(clone_num):
            clones.append(solution[0].__copy__())
        return clones

    def update_progress(self) -> None:
        """ Update the progress after each iteration. """
        self.evaluations += 1
        if self.evaluations % 100 == 0:
            print(f"evaluation {self.evaluations}")
        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = [s[0] for s in self.solutions]
        self.observable.notify_all(**observable_data)

    def get_observable_data(self) -> dict:
        """ Get observable data, with the information that will be send to all observers each time. """
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def get_result(self) -> FloatSolution:
        return self.solutions[0][0]

    def get_name(self) -> str:
        return "CLONALG"

    def draw_history(self):
        for o in range(self.problem.number_of_objectives):
            plt.plot(range(len(self.history)), [s.objectives[o] for s in self.history])
        plt.legend([f"objective {i}" for i in range(self.problem.number_of_objectives)])
        plt.savefig(f"{self.get_name()}_history_{time.time()}.jpg")
        plt.show()
