import random
import threading
import time
from typing import List

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion

from clonal_selection.clonal_selection import ClonalSelection


class ClonalSelectionCognitive(Algorithm[FloatSolution, List[FloatSolution]]):
    def __init__(self,
                 clonal_selections: List[ClonalSelection],
                 mix_rate: float,
                 termination_criterion: TerminationCriterion,
                 evaluator: Evaluator = store.default_evaluator):
        threading.Thread.__init__(self)

        self.clonal_selections = clonal_selections
        self.mix_rate = mix_rate
        self.solutions: List[FloatSolution] = []
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
        solution = []
        for cs in self.clonal_selections:
            solution += cs.create_initial_solutions()
        return solution

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
        for cs in self.clonal_selections:
            cs.step()

        for i in range(self.number_of_populations):
            for j in range(i, self.number_of_populations):
                if random.random() < self.mix_rate:
                    self.mix(self.clonal_selections[i], self.clonal_selections[j])
        solution = []
        for cs in self.clonal_selections:
            solution += cs.solutions
        self.solutions = solution

    def mix(self, clonal_selection_1: ClonalSelection, clonal_selection_2: ClonalSelection):
        position_1 = random.randint(0, len(clonal_selection_1.solutions)-1)
        position_2 = random.randint(0, len(clonal_selection_2.solutions)-1)
        clonal_selection_1.solutions[position_1], clonal_selection_2.solutions[position_2] = \
            clonal_selection_2.solutions[position_2], clonal_selection_1.solutions[position_1]

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
        affinity_values = []
        for cs in self.clonal_selections:
            affinity_values += [(cs.affinity(solution), solution) for solution in cs.solutions]
        result = sorted(affinity_values, key=lambda x: x[0], reverse=True)[0]
        return result[1]

    def get_name(self) -> str:
        return "CLONALG_COGNITIVE"
