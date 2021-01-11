import os
from datetime import datetime

from clonal_selection.clonal_selection import ClonalSelection
from clonal_selection.clonal_selection_cognitive import ClonalSelectionCognitive
from clonal_selection.de_jong_1 import DeJong1
from framework.config import RESULTS_DIR
from framework.runner.multi_algorithm_runner import MultiAlgorithmRunner, save_execution_history
from framework.runner.primitives import ExecutionUnit
from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations


def run() -> None:
    target_path = os.path.join(RESULTS_DIR, f"test_{datetime.now().isoformat()}.json")
    first_execution_unit = ExecutionUnit(
        algorithm_cls=ClonalSelection,
        problem_name="DeJong1"
    ).register_run(
        parameters={
            "problem": DeJong1(-5.12, 5.12, number_of_variables=50),
            "population_size": 200,
            "selection_size": 30,
            "random_cells_number": 50,
            "clone_rate": 20,
            "mutation": PolynomialMutation(probability=1 / 50, distribution_index=20),
            "termination_criterion": StoppingByEvaluations(max_evaluations=500)
        }
    )

    problem_1 = DeJong1(-5.12, 5.12, number_of_variables=50)
    problem_2 = DeJong1(-5.12, 5.12, number_of_variables=50)
    second_execution_unit = ExecutionUnit(
        algorithm_cls=ClonalSelectionCognitive,
        problem_name="DeJong1"
    ).register_run(
        parameters={
            "clonal_selections": [
                ClonalSelection(
                    problem=problem_1,
                    population_size=200,
                    selection_size=30,
                    random_cells_number=50,
                    clone_rate=20,
                    mutation=PolynomialMutation(probability=1 / problem_1.number_of_variables, distribution_index=20),
                ),
                ClonalSelection(
                    problem=problem_2,
                    population_size=200,
                    selection_size=30,
                    random_cells_number=50,
                    clone_rate=20,
                    mutation=PolynomialMutation(probability=2 / problem_2.number_of_variables, distribution_index=20),
                )
            ],
            "mix_rate": 0.4,
            "mixes_number": 2,
            "termination_criterion": StoppingByEvaluations(max_evaluations=500)
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
