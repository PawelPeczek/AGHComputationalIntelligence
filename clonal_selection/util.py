import time
from typing import Type, List

import matplotlib.pyplot as plt
import numpy as np

from jmetal.core.algorithm import Algorithm


def get_mean_solution(results):
    return np.mean([r.variables for r in results], axis=0)


def get_mean_result(results):
    return np.mean([r.objectives[0] for r in results], axis=0)


def get_mean_history(histories):
    return np.mean(histories, axis=0)


def get_std_history(histories):
    return np.std(histories, axis=0)
