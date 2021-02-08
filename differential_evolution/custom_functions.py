from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from abc import abstractmethod
import random


class DEProblem(FloatProblem):
    @abstractmethod
    def affinity(self, solution: FloatSolution):
        pass


class QuadraticFunction(DEProblem):

  def __init__(self, lower_bound, upper_bound):
    super(QuadraticFunction, self).__init__()
    self.number_of_objectives = 1
    self.number_of_variables = 1
    self.number_of_constraints = 0

    self.species_index = -1

    self.upper_bound = [upper_bound] * self.number_of_variables
    self.lower_bound = [lower_bound] * self.number_of_variables

    self.obj_directions = [self.MINIMIZE]

    self.obj_labels = ['y1']

  def evaluate(self, solution: FloatSolution) -> FloatSolution:
    x1 = solution.variables[0]
    y1 = x1 ** 2
    solution.objectives[0] = y1
    return solution

  def create_solution(self) -> FloatSolution:
    new_solution = FloatSolution(self.lower_bound, self.lower_bound,
                                  number_of_objectives=self.number_of_objectives)
    new_solution.variables[0] = random.uniform(self.lower_bound[0], self.upper_bound[0])

    return new_solution

  def affinity(self, solution: FloatSolution) -> float:
    return - solution.variables[0] ** 2

  def get_name(self) -> str:
    return 'Q-F'