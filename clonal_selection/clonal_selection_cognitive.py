import random
import sys
import threading
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedList

from clonal_selection.clonal_selection import ClonalSelection
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion


class ClonalSelectionCognitive(Algorithm[FloatSolution, List[FloatSolution]]):
    def __init__(self,
                 clonal_selections: List[ClonalSelection],
                 mix_rate: float,
                 mixes_number: int,
                 termination_criterion: TerminationCriterion,
                 evaluator: Evaluator = store.default_evaluator):
        threading.Thread.__init__(self)

        self.clonal_selections = clonal_selections
        self.mix_rate = mix_rate
        if mixes_number > len(clonal_selections):
            raise Exception("mixes_number must be lower then number of populations.")
        self.mixes_number = mixes_number
        self.solutions: SortedList[Tuple[FloatSolution, float]] = SortedList([], key=lambda x: -x[1])
        self.evaluations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

        self.evaluator = evaluator
        self.observable = store.default_observable

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        if not self.check_if_problems_ok():
            raise Exception("Only one problem can be solved.")

        self.problem = self.clonal_selections[0].problem
        self.number_of_populations = len(self.clonal_selections)
        self.ranking = dict()
        for i in range(self.number_of_populations):
            self.ranking.update({i: {j: sys.maxsize for j in range(self.number_of_populations)}})

        self.history: List[FloatSolution] = []

    def update_history(self):
        best_solution = self.get_result()
        self.history.append(best_solution)

    def affinity(self, solution: FloatSolution) -> float:
        direction = [-1 if d == self.problem.MINIMIZE else 1 for d in self.problem.obj_directions]
        return np.multiply(direction, self.problem.evaluate(solution).objectives)

    def check_if_problems_ok(self):
        if not self.clonal_selections:
            return False
        problem_type = type(self.clonal_selections[0].problem)
        for cs in self.clonal_selections:
            if type(cs.problem) != problem_type:
                return False
        return True

    def create_initial_solutions(self) -> List[FloatSolution]:
        """ Creates the initial list of solutions of a metaheuristic. """
        solution = SortedList([], key=lambda x: -x[1])
        for cs in self.clonal_selections:
            initial_solutions = cs.create_initial_solutions()
            cs.solutions = initial_solutions
            cs.solutions = cs.evaluate(cs.solutions)
            solution.update(initial_solutions)
        # self.update_history()
        return solution

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
        for cs in self.clonal_selections:
            cs.step()

        for i in range(self.number_of_populations):
            for (j, _) in sorted(self.ranking[i].items(),
                                 key=lambda x: x[1], reverse=True)[:self.mixes_number]:
                if i == j:
                    continue
                if random.random() < self.mix_rate:
                    affinity_i, affinity_j = self.mix(self.clonal_selections[i], self.clonal_selections[j])
                    self.ranking[i][j] = affinity_j
                    self.ranking[j][i] = affinity_i
        solution = SortedList([], key=lambda x: -x[1])
        for cs in self.clonal_selections:
            solution.update(cs.solutions)
        self.solutions = solution
        self.update_history()

    def mix(self, clonal_selection_1: ClonalSelection, clonal_selection_2: ClonalSelection):
        best_solution_1 = clonal_selection_1.solutions[0]
        best_solution_2 = clonal_selection_2.solutions[0]
        worst_solution_1 = clonal_selection_1.solutions[-1]
        worst_solution_2 = clonal_selection_2.solutions[-1]
        clonal_selection_1.solutions.remove(worst_solution_1)
        clonal_selection_1.solutions.add((best_solution_2[0].__copy__(), best_solution_2[1]))
        clonal_selection_2.solutions.remove(worst_solution_2)
        clonal_selection_2.solutions.add((best_solution_1[0].__copy__(), best_solution_1[1]))
        return best_solution_2[1], best_solution_1[1]

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
        return "CLONALG_COGNITIVE"

    def draw_history(self):
        # plt.figure(figsize=(20, 20))
        for o in range(self.problem.number_of_objectives):
            plt.plot(range(len(self.history)), [s.objectives[o] for s in self.history])
        plt.legend([f"objective {i}" for i in range(self.problem.number_of_objectives)])
        plt.savefig(f"{self.get_name()}_history_{time.time()}.jpg")
        plt.show()
