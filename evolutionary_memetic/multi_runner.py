import os

from evolutionary_memetic.memetic import MemeticLocalSearch
from evolutionary_memetic.memetic_cognitive import MemeticCognitiveAlgorithm, Species
from framework.config import RESULTS_DIR
from framework.problems.singleobjective.ackley import Ackley
from framework.runner.multi_algorithm_runner import MultiAlgorithmRunner, save_execution_history
from datetime import datetime

from framework.runner.primitives import ExecutionUnit, DrawingProperties
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
import matplotlib.pyplot as plt

import time

get_results = {}
tries = 0

def draw_function(alg, properties):
    print('\n---\n')
    global get_results
    global tries
    tries += 1

    if properties in get_results:
        get_results[properties].append(alg.get_history())
    else:
        get_results[properties] = [alg.get_history()]

    def mean(x):
        return sum(x)/len(x)

    print(tries)
    if tries == 2:
        for i in get_results:
            plt.plot(list(map(mean, zip(*get_results[i]))), label=i)
            # print('avg {}'.format(i), *map(mean, zip(*get_results[i])))
        plt.legend()
        plt.show()

def draw_comparision_plot(alg1, alg2):
    print('->')
    plt.plot(range(len(alg1.get_history())), alg1.get_history())
    plt.plot(range(len(alg2.get_history())), alg2.get_history())
    plt.legend([alg1.get_name(), alg2.get_name()])
    plt.title(f"Problem {alg1.problem.get_name()} with {alg1.problem.number_of_variables} variables")
    plt.savefig(os.path.join('figs',
                             f"{alg1.problem.get_name()}_{alg1.get_name()}_{alg2.get_name()}_history_{time.time()}.jpg"))
    plt.show()

def run() -> None:
    problem = Ackley(number_of_variables=50)
    mutation =  PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20)
    local_search = MemeticLocalSearch(problem, mutation, StoppingByEvaluations(500))
    max_evaluations = 5000
    target_path = os.path.join(RESULTS_DIR, f"test_{datetime.now().isoformat()}.json")
    first_execution_unit = ExecutionUnit(
        algorithm_cls=MemeticCognitiveAlgorithm,
        problem_name="Ackley",
        drawing_fun=draw_function,
        drawing_series_labels=["RUN1", "RUN1"]
    ).register_run(
        parameters={
            "problem": problem,
            "population_size": 1000,
            "offspring_population_size": 500,
            "mutation": mutation,
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "selection" :BinaryTournamentSelection(),
            "species1": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            "species2": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            "local_search": local_search
        }
    ).register_run(
        parameters={
            "problem": problem,
            "population_size": 1000,
            "offspring_population_size": 5000,
            "mutation": mutation,
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "selection": BinaryTournamentSelection(),
            "species1": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            "species2": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            "local_search": local_search
        }
    )

    second_execution_unit = ExecutionUnit(
        algorithm_cls=MemeticCognitiveAlgorithm,
        problem_name="Ackley",
        drawing_fun=draw_function,
        drawing_series_labels = ["RUN2", "RUN2"]
    ).register_run(
        parameters={
            "problem": problem,
            "population_size": 5000,
            "offspring_population_size": 2000,
            "mutation": mutation,
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "selection": BinaryTournamentSelection(),
            "species1": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            "species2": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            "local_search": local_search
        }
    ).register_run(
        parameters={
            "problem": problem,
            "population_size": 5000,
            "offspring_population_size": 2000,
            "mutation": mutation,
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "selection": BinaryTournamentSelection(),
            "species1": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            "species2": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            "local_search": local_search
        }
    )

    third_execution_unit = ExecutionUnit(
        algorithm_cls=GeneticAlgorithm,
        problem_name="Ackley",
        drawing_fun=draw_function,
        drawing_series_labels=["GENETIC", "GENETIC"]
    ).register_run(
        parameters={
            "problem": problem,
            "population_size": 5000,
            "offspring_population_size": 1000,
            "mutation": mutation,
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "selection": BinaryTournamentSelection(),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    ).register_run(
        parameters={
            "problem": problem,
            "population_size": 5000,
            "offspring_population_size": 1000,
            "mutation": mutation,
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "selection": BinaryTournamentSelection(),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    )

    runner = MultiAlgorithmRunner(
        execution_units=[
            third_execution_unit
        ],
        drawing_properties=
            DrawingProperties(title='Memetic1', target_location=os.path.join(RESULTS_DIR, "photo.png"))
    )
    print("Runner starts evaluation.")
    results = runner.run_all()
    print("Results")
    for run_result in results.run_results:
        print(run_result)
    save_execution_history(execution_history=results, path=target_path)


if __name__ == '__main__':
    run()