import os
from datetime import datetime

from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from differential_evolution.de import DifferentialEvolution
from clonal_selection.de_jong_1 import DeJong1
from clonal_selection.de_jong_5 import DeJong5
from framework.config import RESULTS_DIR
from framework.runner.multi_algorithm_runner import ExecutionUnit, MultiAlgorithmRunner, save_execution_history
from jmetal.problem.singleobjective.unconstrained import Rastrigin


def run() -> None:
    target_path = os.path.join(RESULTS_DIR, "testDE_rastrigin.json")
    rastrigin_problem = Rastrigin(number_of_variables=100)

    variable_species_size_execution_unit = ExecutionUnit(
        algorithm_cls=DifferentialEvolution,
        problem_name="Rastrigin_100dim_variable_species_size"
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 1,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
     ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 20,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        },
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 50,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        },
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 100,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        },
    )

    variable_number_of_species_execution_unit = ExecutionUnit(
        algorithm_cls=DifferentialEvolution,
        problem_name="Rastrigin_100dim_variable_number_of_species"
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "no_species": 1,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "no_species": 10,   # default
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "no_species": 20,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "no_species": 50,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "no_species": 100,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    )

    variable_number_of_partners_execution_unit = ExecutionUnit(
        algorithm_cls=DifferentialEvolution,
        problem_name="Rastrigin_100dim_variable_number_of_partners"
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "number_of_partners": 3,   # default
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "number_of_partners": 5,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "number_of_partners": 10,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "number_of_partners": 15,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 10,
            "number_of_partners": 20,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    )

    variable_iterations_execution_unit = ExecutionUnit(
        algorithm_cls=DifferentialEvolution,
        problem_name="Rastrigin_100dim_variable_max_iter"
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 20,
            "no_species": 50,
            "cr": 0.9, "f": 0.8,
            "max_iter": 250
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 20,
            "no_species": 50,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": rastrigin_problem,
            "each_species_size": 20,
            "no_species": 50,
            "cr": 0.9, "f": 0.8,
            "max_iter": 1000
        }
    )



    runner = MultiAlgorithmRunner(
        execution_units=[
            variable_species_size_execution_unit, variable_number_of_species_execution_unit, variable_number_of_partners_execution_unit, variable_iterations_execution_unit
        ]
    )

    print("Runner starts evaluation.")
    results = runner.run_all()
    print("Results")
    for run_result in results.run_results:
        print({
            "problem_name": run_result.problem_name,
            "algorithm": run_result.algorithm_name,
            "fitness": run_result.fitness
        })
    save_execution_history(execution_history=results, path=target_path)


if __name__ == '__main__':
    run()
