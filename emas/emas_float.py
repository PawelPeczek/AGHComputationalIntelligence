from framework.problems.singleobjective.ackley import Ackley
from framework.problems.singleobjective.griewank import Griewank
from jmetal.algorithm.singleobjective import EvolutionStrategy
from jmetal.algorithm.singleobjective.emas import Emas
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.operator.crossover import DiscreteCrossover
from jmetal.operator.death import ThresholdDeath
from jmetal.operator.energy_exchange import FractionEnergyExchange
from jmetal.operator.neighbours import RandomNeighbours
from jmetal.operator.reproduction import FractionEnergyReproduction, EmasSpeciesReproduction
from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    for var in [10, 50, 100]:
        print(f'============================================RASTRIGIN VAR: {var}=================================================')
        problem = Ackley(number_of_variables=var)
        evaluations = 25000

        evolution_strategy = EvolutionStrategy(
            problem=problem,
            mu=10,
            lambda_=10,
            elitist=True,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=evaluations)
        )


        evolution_strategy.run()
        result = evolution_strategy.get_result()

        df = pd.DataFrame(data=list(zip(evolution_strategy.evaluations_history,
                                        evolution_strategy.fitness_history,
                                        len(evolution_strategy.evaluations_history) * ['es_mu10_lambd_10'] ) ),
                          columns=['evaluations', 'fitness', 'algorithm'])


        meta_results_df = pd.DataFrame(data=np.array([result.variables[0], result.objectives[0], evolution_strategy.total_computing_time, 'es_mu10_lambd_10']).reshape(1, -1),
                                       columns=('final_solution', 'final_fitness', 'computing_time', 'algorithm'))


        individual_energy = 10

        initial_population_sizes = np.arange(10, 150, 50)
        reproduction_thresholds = np.linspace(0.5*individual_energy, 1.5*individual_energy, 3)
        energy_exchange_fractions = np.linspace(0.25, 0.75, 3)
        death_thresholds = np.linspace(0.5*individual_energy, 1.5*individual_energy, 3)
        reproduction_energy_fractions = np.linspace(0.25, 0.75, 3)

        for ips in initial_population_sizes:
            for rt in reproduction_thresholds:
                for eef in energy_exchange_fractions:
                    for dt in death_thresholds:
                        for ref in reproduction_energy_fractions:
                            print(f'initial_population_size: {ips}')
                            print(f'reproduction_threshold: {rt}')
                            print(f'energy_exchange_fraction: {eef}')
                            print(f'death_threshold: {dt}')
                            print(f'reproduction_energy_fraction: {ref}')
                            algorithm = Emas(
                                problem=problem,
                                initial_population_size=ips,
                                initial_inidividual_energy=individual_energy,
                                reproduction_threshold=rt,
                                energy_exchange_operator=FractionEnergyExchange(eef),
                                death_operator=ThresholdDeath(threshold=dt, neighbours_operator=RandomNeighbours()),
                                termination_criterion=StoppingByEvaluations(max_evaluations=evaluations),
                                neighbours_operator=RandomNeighbours(),
                                reproduction_operator=FractionEnergyReproduction(ref,
                                                                                 PolynomialMutation(1.0 / problem.number_of_variables),
                                                                                 SBXCrossover(0.5)),
                                no_species=1,
                                species_size=ips
                            )
                            try:
                                algorithm.run()
                                result = algorithm.get_result()

                                algo_df = pd.DataFrame(data=list(zip(algorithm.evaluations_history,
                                                                algorithm.fitness_history,
                                                                len(algorithm.evaluations_history) * [f'emas_ips_{ips}_rt_{rt}_eef_{eef}_dt_{dt}_ref_{ref}'],
                                                                len(algorithm.evaluations_history) * [ips],
                                                                len(algorithm.evaluations_history) * [rt],
                                                                len(algorithm.evaluations_history) * [eef],
                                                                len(algorithm.evaluations_history) * [dt],
                                                                len(algorithm.evaluations_history) * [ref])),
                                                  columns=['evaluations', 'fitness', 'algorithm', 'initial_population_size', 'reproduction_threshold', 'energy_exchange_fraction', 'death_threshold', 'reproduction_energy_fraction'])

                                df = df.append(algo_df)

                                meta_results_df = meta_results_df.append(pd.DataFrame(data=np.array(
                                    [result.variables[0], result.objectives[0], algorithm.total_computing_time,
                                     f'emas_ips_{ips}_rt_{rt}_eef_{eef}_dt_{dt}_ref_{ref}', ips, rt, eef, dt, ref]).reshape(1, -1),
                                                               columns=('final_solution', 'final_fitness', 'computing_time',
                                                                        'algorithm', 'initial_population_size', 'reproduction_threshold', 'energy_exchange_fraction', 'death_threshold', 'reproduction_energy_fraction')))

                            except Exception as e:
                                print(f'Exception catched: {e}')
                                print(ips, rt, eef, dt, ref)
                                continue


        meta_results_df.final_fitness = meta_results_df.final_fitness.astype(float)
        meta_results_df.final_solution = meta_results_df.final_solution.astype(float)
        meta_results_df.computing_time = meta_results_df.computing_time.astype(float)

        meta_results_df.to_csv(f'ackley_{var}_meta_results.csv')
        df.to_csv(f'ackley{var}_results.csv')

    # sns.lineplot(x='evaluations', y='fitness', hue='algorithm', data=df)
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()



    # sns.barplot(x='algorithm', y='final_fitness', data=meta_results_df)
    # plt.show()

    # algorithm = Emas(
    #     problem=problem,
    #     initial_population_size=40,
    #     initial_inidividual_energy=10,
    #     reproduction_threshold=12,
    #     energy_exchange_operator=FractionEnergyExchange(0.25),
    #     death_operator=ThresholdDeath(threshold=7, neighbours_operator=RandomNeighbours()),
    #     termination_criterion=StoppingByEvaluations(max_evaluations=evaluations),
    #     neighbours_operator=RandomNeighbours(),
    #     reproduction_operator=EmasSpeciesReproduction(0.25, PolynomialMutation(1.0 / problem.number_of_variables)),
    #     no_species=4,
    #     species_size=10
    # )

    # emas.run()

    # df = pd.DataFrame(data=list(zip(emas.evaluations_history,
    #                                 emas.fitness_history,
    #                                 len(emas.evaluations_history) * ['emas'] ) ),
    #                   columns=['evaluations', 'fitness', 'algorithm'])
