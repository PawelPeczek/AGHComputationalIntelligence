import os
from datetime import datetime

from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from clonal_selection.clonal_selection import ClonalSelection
from clonal_selection.clonal_selection_cognitive import ClonalSelectionCognitive
from clonal_selection.de_jong_1 import DeJong1
from clonal_selection.de_jong_5 import DeJong5
from framework.config import RESULTS_DIR
from framework.runner.multi_algorithm_runner import ExecutionUnit, \
    MultiAlgorithmRunner, save_execution_history


def run() -> None:
    target_path = os.path.join(RESULTS_DIR, f"test_{datetime.now().isoformat()}.json")
    first_execution_unit = ExecutionUnit(
        algorithm_cls=ClonalSelection,
        problem_name="DeJong5"
    ).register_run(
        parameters={
            "problem": DeJong1(-5.12, 5.12),
            "population_size": 100, "selection_size": 30,
            "mutation": PolynomialMutation(probability=1.0 / 2, distribution_index=20),
            "termination_criterion": StoppingByEvaluations(max_evaluations=500)
        }
    ).register_run(
        parameters={
            "problem": DeJong1(-5.12, 5.12),
            "population_size": 80, "selection_size": 30,
            "mutation": PolynomialMutation(probability=1.0 / 2, distribution_index=20),
            "termination_criterion": StoppingByEvaluations(max_evaluations=500)
        },
    )
    second_execution_unit = ExecutionUnit(
        algorithm_cls=ClonalSelection,
        problem_name="DeJong5"
    ).register_run(
        parameters={
            "problem": DeJong5(-5, 5),
            "population_size": 100, "selection_size": 30,
            "mutation": PolynomialMutation(probability=1.0 / 2, distribution_index=20),
            "termination_criterion": StoppingByEvaluations(max_evaluations=500)
        }
    )
    problem = DeJong1(-5.12, 5.12)
    third_execution_unit = ExecutionUnit(
        algorithm_cls=ClonalSelectionCognitive,
        problem_name="DeJong1"
    ).register_run(
        parameters={
            "clonal_selections": [
                ClonalSelection(
                    problem=problem,
                    population_size=100,
                    selection_size=30,
                    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
                    termination_criterion=StoppingByEvaluations(max_evaluations=2000)
                ),
                ClonalSelection(
                    problem=problem,
                    population_size=100,
                    selection_size=30,
                    mutation=PolynomialMutation(probability=1.0 / (problem.number_of_variables * 2),
                                                distribution_index=20),
                    termination_criterion=StoppingByEvaluations(max_evaluations=2000)
                )
            ],
            "mix_rate": 0.4,
            "mixes_number": 2,
            "termination_criterion": StoppingByEvaluations(max_evaluations=2000)
        }
    )
    problem = DeJong1(-5.12, 5.12)
    fourth_execution_unit = ExecutionUnit(
        algorithm_cls=ClonalSelectionCognitive,
        problem_name="DeJong1"
    ).register_run(
        parameters={
            "clonal_selections": [
                ClonalSelection(
                    problem=problem,
                    population_size=100,
                    selection_size=30,
                    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
                    termination_criterion=StoppingByEvaluations(max_evaluations=2000)
                ),
                ClonalSelection(
                    problem=problem,
                    population_size=100,
                    selection_size=30,
                    mutation=PolynomialMutation(probability=1.0 / (problem.number_of_variables * 2),
                                                distribution_index=20),
                    termination_criterion=StoppingByEvaluations(max_evaluations=2000)
                )
            ],
            "mix_rate": 0.4,
            "mixes_number": 1,
            "termination_criterion": StoppingByEvaluations(max_evaluations=2000)
        }
    )
    runner = MultiAlgorithmRunner(
        execution_units=[
            first_execution_unit, second_execution_unit, third_execution_unit, fourth_execution_unit
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
