import matplotlib.pyplot as plt
import time
import os


def draw_comparision_plot(alg1, alg2):
    plt.plot(range(len(alg1.get_history())), alg1.get_history())
    plt.plot(range(len(alg2.get_history())), alg2.get_history())
    plt.legend([alg1.get_name(), alg2.get_name()])
    plt.title(f"Problem {alg1.problem.get_name()} with {alg1.problem.number_of_variables} variables")
    plt.savefig(os.path.join('figs', f"{alg1.problem.get_name()}_{alg1.get_name()}_{alg2.get_name()}_history_{time.time()}.jpg"))
    plt.show()
