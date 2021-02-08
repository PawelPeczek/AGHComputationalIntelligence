from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.generator import Generator, RandomGenerator
from jmetal.util.evaluator import Evaluator, SequentialEvaluator
from jmetal.operator.mutation import Mutation
from jmetal.core.problem import Problem


# classical Evolution Strategy with history logging
class EvolutionStrategyWithHistory(EvolutionStrategy):
    def __init__(self,
                problem: Problem,
                mu: int,
                lambda_: int,
                elitist: bool,
                mutation: Mutation,
                termination_criterion: TerminationCriterion,
                population_generator: Generator = RandomGenerator(),
                population_evaluator: Evaluator = SequentialEvaluator()):
        super(EvolutionStrategyWithHistory, self).__init__(
            problem=problem,
            mu=mu,
            lambda_=lambda_,
            elitist=elitist,
            mutation=mutation,
            termination_criterion=termination_criterion,
            population_generator=population_generator,
            population_evaluator=population_evaluator)
        self.history = []

    def step(self):
        super().step()
        self.update_history()

    def update_history(self):
        best_fitness = self.solutions[0].objectives[0]
        self.history.append(best_fitness)

    def get_history(self):
        return self.history

    def get_name(self):
        return 'NormalEvolutionStrategy'
