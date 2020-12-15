from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy
from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.operator import Mutation
from jmetal.core.solution import FloatSolution
from jmetal.util.ckecking import Check
import random

class CognitivePolynomialMutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, distribution_index: float = 0.20,
    look_at_others_probability: float = 0.2):
        super(CognitivePolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index
        self.look_at_others_probability = look_at_others_probability
        self.follow_solutions = []
        self.steps = 0
        self.avg = 0
        self.avg_counter = 0

    def execute(self, solution: FloatSolution) -> FloatSolution:
        Check.that(issubclass(type(solution), FloatSolution), "Solution type invalid")
        copied_solution = solution.__copy__()
        if copied_solution not in self.follow_solutions:
            self.follow_solutions.append(copied_solution)
        self.follow_solutions.sort(key=lambda x: x.objectives[0], reverse=False)
        self.follow_solutions = self.follow_solutions[:140]

        if (self.avg_counter % 140 == 0):
            self.avg_counter = 0

        self.avg = (self.avg * self.avg_counter + copied_solution.objectives[0]) / (self.avg_counter + 1)
        self.avg_counter += 1
        self.steps += 1

        vector = self.get_follow_vector()

        for i in range(solution.number_of_variables):
            rand = random.random()

            if rand <= self.probability or self.steps < 140:
                y = solution.variables[i]
                yl, yu = solution.lower_bound[i], solution.upper_bound[i]

                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    rnd = random.random()
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = pow(val, mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = 1.0 - pow(val, mut_pow)

                    y += deltaq * (yu - yl)
                    if y < solution.lower_bound[i]:
                        y = solution.lower_bound[i]
                    if y > solution.upper_bound[i]:
                        y = solution.upper_bound[i]

                solution.variables[i] = y
            elif rand <= self.probability + self.look_at_others_probability:
                diff = vector[i] - solution.variables[i]
                calculated_value = solution.variables[i] + 0.1 * diff
                solution.variables[i] = max(min(calculated_value,solution.upper_bound[i]),solution.lower_bound[i])

        return solution
    
    # fitness -> 0, m -> 1
    # fitness -> inf, m -> -0.001
    def get_factor_on_fitness(self, fitness, avg_fitness):
        if (fitness <= avg_fitness):
            return 1 - (fitness / avg_fitness)
        else:
            return 0 * (fitness / avg_fitness)

    def get_follow_vector(self):
        avg_fitness = self.avg
        no_of_variables = self.follow_solutions[0].number_of_variables
        vector = [0] * no_of_variables
        total_factor = 0

        for sol in self.follow_solutions:
            fitness = sol.objectives[0]
            factor = self.get_factor_on_fitness(fitness, avg_fitness)
            total_factor += abs(factor)

            for i in range(no_of_variables):
                vector[i] += factor * sol.variables[i]
        
        for i in range(len(vector)):
            if (total_factor):
                vector[i] /= total_factor

        return vector

    def get_name(self):
        return 'Cognitive polynomial mutation'
