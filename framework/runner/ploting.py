from typing import List

import matplotlib.pyplot as plt

from framework.runner.primitives import DrawingProperties, PlotSeries


def plot_algorithm_results(
    drawing_properties: DrawingProperties,
    series: List[PlotSeries]
) -> None:
    plt.figure(figsize=drawing_properties.fig_size)
    for s in series:
        plt.plot(s.xs, s.ys, label=s.label)
    plt.title(drawing_properties.title)
    plt.xlabel(drawing_properties.x_label)
    plt.ylabel(drawing_properties.y_label)
    plt.legend()
    if drawing_properties.target_location is not None:
        plt.savefig(drawing_properties.target_location)
    if drawing_properties.verbose:
        plt.show()
