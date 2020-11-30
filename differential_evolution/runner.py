import os
from datetime import datetime

from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from differential_evolution.de import DifferentialEvolution
from clonal_selection.de_jong_1 import DeJong1
from clonal_selection.de_jong_5 import DeJong5
from framework.config import RESULTS_DIR
from framework.runner.multi_algorithm_runner import ExecutionUnit, MultiAlgorithmRunner, save_execution_history


def run() -> None:
    target_path = os.path.join(RESULTS_DIR, "testDE.json")

    first_execution_unit = ExecutionUnit(
        algorithm_cls=DifferentialEvolution,
        problem_name="DeJong1"
    ).register_run(
        parameters={
            "problem": DeJong1(-5.12, 5.12),
            "each_species_size": 10,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    ).register_run(
        parameters={
            "problem": DeJong1(-5.12, 5.12),
            "each_species_size": 10,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        },
    )

    second_execution_unit = ExecutionUnit(
        algorithm_cls=DifferentialEvolution,
        problem_name="DeJong5"
    ).register_run(
        parameters={
            "problem": DeJong5(-5, 5),
            "each_species_size": 10,
            "cr": 0.9, "f": 0.8,
            "max_iter": 500
        }
    )

    runner = MultiAlgorithmRunner(
        execution_units=[
            first_execution_unit, second_execution_unit
        ]
    )

    print("Runner starts evaluation.")
    results = runner.run_all()
    print("Results")
    for run_result in results.run_results:
        print(run_result)
    save_execution_history(execution_history=results, path=target_path)


if __name__ == '__main__':
    run()
