from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from framework.runner.primitives import DrawingProperties, PlotSeries


def plot_algorithm_results(
    drawing_properties: DrawingProperties,
    series: List[List[Optional[PlotSeries]]]
) -> None:
    plt.figure(figsize=drawing_properties.fig_size)
    for s in series:
        plot_average_series_results(series_results=s)
    plt.title(drawing_properties.title)
    plt.xlabel(drawing_properties.x_label)
    plt.ylabel(drawing_properties.y_label)
    plt.legend()
    if drawing_properties.target_location is not None:
        plt.savefig(drawing_properties.target_location)
    if drawing_properties.verbose:
        plt.show()


def plot_average_series_results(
    series_results: List[Optional[PlotSeries]]
) -> None:
    xs = np.array([p.xs for p in series_results if p is not None]).mean(axis=0)
    all_ys = np.array([p.ys for p in series_results if p is not None])
    label = [p.label for p in series_results if p is not None][0]
    ys = np.mean(all_ys, axis=0)
    if all_ys.shape[0] > 1:
        y_err = np.std(all_ys, axis=0)
        _, _, bars = plt.errorbar(xs, ys, label=label, yerr=y_err)
        [bar.set_alpha(0.35) for bar in bars]
    else:
        plt.plot(xs, ys, label=label)
