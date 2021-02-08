import os
from datetime import datetime

from jmetal.operator import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from differential_evolution.de import DifferentialEvolution
from framework.config import RESULTS_DIR
from framework.runner.multi_algorithm_runner import ExecutionUnit, MultiAlgorithmRunner, save_execution_history
from framework.problems.singleobjective.griewank import Griewank

def get_execution_unit(number_of_partners, iters) -> ExecutionUnit:
    execution_unit = ExecutionUnit(
        algorithm_cls=DifferentialEvolution,
        problem_name= f"Griewank_100dim_{number_of_partners}_partners"
    )

    griewank_problem = Griewank(number_of_variables=100, lower_bound=-100, upper_bound=100)
    
    for _ in range(iters):
        execution_unit.register_run(
            parameters={
            "problem": griewank_problem,
            "each_species_size": 50,
            "number_of_partners": number_of_partners,
            "cr": 0.9, "f": 0.8,
            "max_iter": 250
            }
        )
    
    return execution_unit

def run() -> None:
    target_path = os.path.join(RESULTS_DIR, "testDE_Griewank.json")

    execution_unit_3_partners = get_execution_unit(3, 1)
    execution_unit_5_partners = get_execution_unit(5, 1)
    execution_unit_7_partners = get_execution_unit(7, 1)
    execution_unit_10_partners = get_execution_unit(10, 1)



    runner = MultiAlgorithmRunner(
        execution_units=[
            execution_unit_3_partners,
            execution_unit_5_partners,
            execution_unit_7_partners,
            execution_unit_10_partners
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
