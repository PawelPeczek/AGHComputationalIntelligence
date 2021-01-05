from species import StrategyParams, SocioCognitiveEvolutionStrategy
from jmetal.util.termination_criterion import StoppingByEvaluations
from plot import draw_comparision_plot
from evolution_strategy import EvolutionStrategyWithHistory
from jmetal.operator import PolynomialMutation

# problems
from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin

if __name__ == '__main__':

    problem = Rastrigin(number_of_variables=50)
    mu = 20
    lambda_ = 140
    elitist = True
    max_evaluations = 25000

    strategies_params = [
        StrategyParams(mu=mu, lambda_=lambda_, elitist=elitist, look_at_others_probability=0.2)
        # StrategyParams(mu=1, lambda_=5, elitist=True, look_at_others_probability=0.2)
    ]

    algorithm = SocioCognitiveEvolutionStrategy(
        problem=problem,
        strategies_params=strategies_params,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm_normal = EvolutionStrategyWithHistory(
        problem=problem,
        mu=mu,
        lambda_=lambda_,
        elitist=elitist,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.run()
    result = algorithm.get_result()
    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(result.variables[0]))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))
    print(result.variables)

    algorithm_normal.run()

    draw_comparision_plot(algorithm, algorithm_normal)
