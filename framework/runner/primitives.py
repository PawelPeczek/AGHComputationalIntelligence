from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Tuple, Optional, Union, List, Type, Callable

import numpy as np
from dataclasses_json import dataclass_json

from jmetal.core.algorithm import Algorithm, S, R


@dataclass(frozen=True)
class DrawingProperties:
    title: str
    x_label: str = "Iterations"
    y_label: str = "Fitness"
    fig_size: Tuple[int, int] = (16, 9)
    target_location: Optional[str] = None
    verbose: bool = True


@dataclass(frozen=True)
class PlotSeries:
    xs: Union[np.ndarray, List[Union[int, float]]]
    ys: Union[np.ndarray, List[Union[int, float]]]
    label: str


@dataclass(frozen=True)
class ExecutionUnit:
    problem_name: str
    algorithm_cls: Type[Algorithm[S, R]]
    drawing_fun: Optional[Callable[[Algorithm[S, R], str], PlotSeries]] = None
    run_parameters: List[dict] = field(default_factory=list)
    drawing_series_labels: List[str] = field(default_factory=list)

    def register_run(
        self,
        parameters: dict,
        drawing_label: Optional[str] = None
    ) -> ExecutionUnit:
        new_run_parameters = self.run_parameters
        new_run_parameters.append(parameters)
        drawing_series_labels = self.drawing_series_labels
        drawing_series_labels.append(
            drawing_label if drawing_label is not None
            else f"run_{len(new_run_parameters)}"
        )
        return replace(
            self,
            run_parameters=new_run_parameters,
            drawing_series_labels=drawing_series_labels
        )


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