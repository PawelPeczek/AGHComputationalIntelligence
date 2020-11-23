from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Type, List, Optional

from dataclasses_json import dataclass_json
from jmetal.core.algorithm import Algorithm, S, R
from tqdm import tqdm


@dataclass(frozen=True)
class ExecutionUnit:
    problem_name: str
    algorithm_cls: Type[Algorithm[S, R]]
    run_parameters: List[dict] = field(default_factory=list)

    def register_run(
        self,
        parameters: dict
    ) -> ExecutionUnit:
        new_run_parameters = self.run_parameters
        new_run_parameters.append(parameters)
        return replace(self, run_parameters=new_run_parameters)


@dataclass_json
@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    algorithm_name: str
    problem_name: str
    solution: Optional[R] = None
    fitness: Optional[List[float]] = None
    execution_time: Optional[float] = None
    error_description: Optional[str] = None


@dataclass_json
@dataclass(frozen=True)
class ExecutionHistory:
    run_results: List[ExecutionResult] = field(default_factory=list)

    def register_run_results(
        self,
        results: List[ExecutionResult]
    ) -> ExecutionHistory:
        new_run_results = self.run_results
        new_run_results.extend(results)
        return replace(self, run_results=new_run_results)


def save_execution_history(
    execution_history: ExecutionHistory,
    path: str
) -> None:
    with open(path, "w") as f:
        json.dump(execution_history.to_dict(), f, indent=4)


class MultiAlgorithmRunner:

    def __init__(
        self,
        execution_units: Optional[List[ExecutionUnit]] = None
    ):
        if execution_units is None:
            self.__execution_units: List[ExecutionUnit] = []
        else:
            self.__execution_units = execution_units

    def register_running_unit(self, running_unit: ExecutionUnit) -> None:
        self.__execution_units.append(running_unit)

    def run_all(self) -> ExecutionHistory:
        execution_history = ExecutionHistory()
        for execution_unit in tqdm(self.__execution_units):
            run_results = self.__run_execution_unit(execution_unit=execution_unit)
            execution_history = execution_history.register_run_results(
                results=run_results
            )
        return execution_history

    def __run_execution_unit(
        self,
        execution_unit: ExecutionUnit
    ) -> List[ExecutionResult]:
        run_results = []
        for parameters in execution_unit.run_parameters:
            run_result = self.__execute_algorithm_with_given_parameters(
                algorithm_cls=execution_unit.algorithm_cls,
                problem_name=execution_unit.problem_name,
                parameters=parameters
            )
            run_results.append(run_result)
        return run_results

    def __execute_algorithm_with_given_parameters(
        self,
        algorithm_cls: Type[Algorithm],
        problem_name: str,
        parameters: dict
    ) -> ExecutionResult:
        try:
            algorithm = algorithm_cls(**parameters)
            algorithm.run()
            result = algorithm.get_result()
            return ExecutionResult(
                success=True,
                algorithm_name=algorithm.get_name(),
                problem_name=problem_name,
                solution=result.variables,
                fitness=result.objectives,
                execution_time=algorithm.total_computing_time
            )
        except Exception as e:
            error_description = f"Error of type: {type(e)} occurred. Cause: {e}"
            return ExecutionResult(
                success=False,
                algorithm_name=algorithm_cls.__name__,
                problem_name=problem_name,
                error_description=error_description
            )

