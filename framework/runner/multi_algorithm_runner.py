from __future__ import annotations

import json
from typing import Type, List, Optional, Tuple

from framework.runner.ploting import plot_algorithm_results
from framework.runner.primitives import ExecutionUnit, ExecutionResult, \
    ExecutionHistory, DrawingProperties, PlotSeries
from jmetal.core.algorithm import Algorithm
from tqdm import tqdm


def save_execution_history(
    execution_history: ExecutionHistory,
    path: str
) -> None:
    with open(path, "w") as f:
        json.dump(execution_history.to_dict(), f, indent=4)


class MultiAlgorithmRunner:

    def __init__(
        self,
        execution_units: Optional[List[ExecutionUnit]] = None,
        drawing_properties: Optional[DrawingProperties] = None
    ):
        if execution_units is None:
            self.__execution_units: List[ExecutionUnit] = []
        else:
            self.__execution_units = execution_units
        self.__drawing_properties = drawing_properties

    def register_running_unit(self, running_unit: ExecutionUnit) -> None:
        self.__execution_units.append(running_unit)

    def run_all(self) -> ExecutionHistory:
        execution_history = ExecutionHistory()
        global_plot_series = []
        for execution_unit in tqdm(self.__execution_units):
            run_results, plot_series = self.__run_execution_unit(execution_unit=execution_unit)
            execution_history = execution_history.register_run_results(
                results=run_results
            )
            global_plot_series.extend(plot_series)
        if self.__drawing_properties is not None and len(global_plot_series) > 0:
            plot_algorithm_results(
                drawing_properties=self.__drawing_properties,
                series=global_plot_series
            )
        return execution_history

    def __run_execution_unit(
        self,
        execution_unit: ExecutionUnit
    ) -> Tuple[List[ExecutionResult], List[Optional[PlotSeries]]]:
        run_results, plot_series = [], []
        for i in range(len(execution_unit.run_parameters)):
            run_result, plot_serie = self.__execute_algorithm_with_given_parameters(
                execution_unit=execution_unit,
                execution_id=i
            )
            run_results.append(run_result)
            if plot_serie is not None:
                plot_series.append(plot_serie)
        return run_results, plot_series

    def __execute_algorithm_with_given_parameters(
        self,
        execution_unit: ExecutionUnit,
        execution_id: int
    ) -> Tuple[ExecutionResult, Optional[PlotSeries]]:
        try:
            algorithm = execution_unit.algorithm_cls(
                **execution_unit.run_parameters[execution_id]
            )
            algorithm.run()
            result = algorithm.get_result()
            plot_series = execution_unit.drawing_fun(
                algorithm, execution_unit.drawing_series_labels[execution_id]
            ) if execution_unit.drawing_fun is not None else None
            return ExecutionResult(
                success=True,
                algorithm_name=algorithm.get_name(),
                problem_name=execution_unit.problem_name,
                solution=result.variables,
                fitness=result.objectives,
                execution_time=algorithm.total_computing_time
            ), plot_series
        except Exception as e:
            error_description = f"Error of type: {type(e)} occurred. Cause: {e}"
            return ExecutionResult(
                success=False,
                algorithm_name=execution_unit.algorithm_cls.__name__,
                problem_name=execution_unit.problem_name,
                error_description=error_description
            ), None

