import os

from evolutionary_memetic.memetic import MemeticLocalSearch
from evolutionary_memetic.memetic_cognitive import MemeticCognitiveAlgorithm, Species
from framework.config import RESULTS_DIR
from framework.problems.singleobjective.ackley import Ackley
from framework.problems.singleobjective.griewank import Griewank
from framework.problems.singleobjective.schafferF7 import SchafferF7
from framework.problems.singleobjective.schwefel import Schwefel
from framework.runner.multi_algorithm_runner import MultiAlgorithmRunner, save_execution_history
from datetime import datetime

from framework.runner.primitives import ExecutionUnit, DrawingProperties
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations
import matplotlib.pyplot as plt


class DrawingClass:

    def __init__(self, registered_runs):
        self.res = {}
        self.tries = 0
        self.registered_runs = registered_runs

    def draw_avg_function(self, alg, properties):
        self.tries += 1
        if properties in self.res:
            self.res[properties].append(alg)
        else:
            self.res[properties] = [alg]

        def mean(x):
            return sum(x) / len(x)

        if self.tries == self.registered_runs:
            for i in self.res:
                tmp = []
                for j in self.res[i]:
                    for o in range(j.problem.number_of_objectives):
                        tmp.append([s.objectives[o] for s in j.history[:400]])
                self.res[i] = tmp

            for i in self.res:
                plt.plot(list(map(mean, zip(*self.res[i]))), label=i)

            plt.title('{} {}'.format(alg.problem.get_name(), alg.problem.number_of_variables))
            plt.legend()
            plt.savefig(
                os.path.join(RESULTS_DIR, '{} {}.png'.format(alg.problem.get_name(), alg.problem.number_of_variables)))
            plt.show()


def run() -> None:
    problem = Ackley(number_of_variables=150)
    # problem = Griewank(number_of_variables=150)
    # problem = Schwefel(number_of_variables=150)
    # problem = SchafferF7(number_of_variables=150)
    mutation = PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20)
    local_search = MemeticLocalSearch(problem, mutation, StoppingByEvaluations(250))
    local_search2 = MemeticLocalSearch(problem, mutation, StoppingByEvaluations(500))

    max_evaluations = 1000000

    drawing_class = DrawingClass(registered_runs=6)

    target_path = os.path.join(RESULTS_DIR, f"test_{datetime.now().isoformat()}.json")
    first_execution_unit = ExecutionUnit(
        algorithm_cls=MemeticCognitiveAlgorithm,
        problem_name="Ackley",
        drawing_fun=drawing_class.draw_avg_function,
        drawing_series_labels=["RUN1", "RUN1"]
    ).register_run(
        parameters={
            "problem": problem,
            "population_size": 5000,
            "offspring_population_size": 1000,
            "mutation": mutation,
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "selection": BinaryTournamentSelection(),
            "species1": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=1000)
            ),
            "species2": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=1000)
            ),
            "local_search": local_search,
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
            "species1": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=1000)
            ),
            "species2": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=1000)
            ),
            "local_search": local_search,
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    )

    second_execution_unit = ExecutionUnit(
        algorithm_cls=MemeticCognitiveAlgorithm,
        problem_name="Ackley",
        drawing_fun=drawing_class.draw_avg_function,
        drawing_series_labels=["RUN2", "RUN2"]
    ).register_run(
        parameters={
            "problem": problem,
            "population_size": 5000,
            "offspring_population_size": 2500,
            "mutation": mutation,
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "selection": BinaryTournamentSelection(),
            "species1": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=1000)
            ),
            "species2": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search2,
                termination_criterion=StoppingByEvaluations(max_evaluations=1000)
            ),
            "local_search": local_search,
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    ).register_run(
        parameters={
            "problem": problem,
            "population_size": 5000,
            "offspring_population_size": 2500,
            "mutation": mutation,
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "selection": BinaryTournamentSelection(),
            "species1": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search,
                termination_criterion=StoppingByEvaluations(max_evaluations=1000)
            ),
            "species2": Species(
                mutation=mutation,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                selection=BinaryTournamentSelection(),
                local_search=local_search2,
                termination_criterion=StoppingByEvaluations(max_evaluations=1000)
            ),
            "local_search": local_search,
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    )

    # modify genetic to get history (see MemeticCognitiveAlgorithm)
    third_execution_unit = ExecutionUnit(
        algorithm_cls=GeneticAlgorithm,
        problem_name="Ackley",
        drawing_fun=drawing_class.draw_avg_function,
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
            first_execution_unit, second_execution_unit, third_execution_unit
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
