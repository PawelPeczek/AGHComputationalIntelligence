from species import StrategyParams, SocioCognitiveEvolutionStrategy
from cognitive_mutation import CognitivePolynomialMutation
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':

    problem = Sphere(number_of_variables=20)

    strategies_params = [
        StrategyParams(mu=20, lambda_=140, elitist=True, look_at_others_probability=0.2),
        StrategyParams(mu=1, lambda_=5, elitist=True, look_at_others_probability=0.2)
    ]

    algorithm = SocioCognitiveEvolutionStrategy(
        problem=problem,
        strategies_params=strategies_params,
        termination_criterion=StoppingByEvaluations(max_evaluations=25000)
    )

    # algorithm = EvolutionStrategy(
    #     problem=problem,
    #     mu=20,
    #     lambda_=140,
    #     elitist=False,
    #     # mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables),
    #     mutation=CognitivePolynomialMutation(probability=1.0 / problem.number_of_variables, look_at_others_probability=0.2),
    #     termination_criterion=StoppingByEvaluations(max_evaluations=20000)
    # )

    if(False):
        N = 30
        Fitness = 0

        for i in range(N):
            algorithm.run()
            result = algorithm.get_result()
            Fitness += result.objectives[0]

        print('Fitness:  ' + str(Fitness / N))
    else:
        algorithm.run()
        result = algorithm.get_result()
        print('Algorithm: ' + algorithm.get_name())
        print('Problem: ' + problem.get_name())
        print('Solution: ' + str(result.variables[0]))
        print('Fitness:  ' + str(result.objectives[0]))
        print('Computing time: ' + str(algorithm.total_computing_time))
        print(result.variables)

    ### RESULTS
    # base: 0.26340222298954524
    # going near best: 0.008740322342651355
    # going far from worst: 0.07485124269069315
