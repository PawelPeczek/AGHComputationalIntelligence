#!/usr/bin/env python
# coding: utf-8

# In[3]:
import os
import sys

sys.path.append("../../")

# In[4]:


from clonal_selection.clonal_selection_anti_worse_pro_elite import ClonalSelectionAntiWorseProElite
from clonal_selection.util import get_mean_solution, get_mean_result, get_mean_history
from framework.problems.singleobjective.schwefel import Schwefel
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.operator import PolynomialMutation, SimpleRandomMutation

# In[5]:


import itertools
import json

# # Clonal Selection Elite

# In[6]:


number_of_variables = [50, 100, 200]  # 500?
population_size = [100, 200]
selection_size = [2 / 20, 5 / 20]
push_pull_random = [(1 / 3, 1 / 3, 1 / 3), (1 / 2, 1 / 2, 0)]
mutation_probability = [1, 3]
clone_rate = [1 / 20, 2 / 20]
random_cells_number = [2 / 20, 5 / 20]

grid = [number_of_variables, population_size, selection_size, push_pull_random, mutation_probability, clone_rate,
        random_cells_number]

grid = list(itertools.product(*grid))
# pprint(grid[:5])
print(len(grid))

# In[10]:


max_evaluations = 800
number_of_tries = 10

for n, ps, ss, (pull, push, random), mp, cr, rcn in grid:
    problem = Schwefel(number_of_variables=n, lower_bound=-500, upper_bound=500)
    file_name = f'results_elite/clonal_selection_elite_{problem.get_name()}_{n}_{ps}_{ss}_{pull}_{push}_{random}_{mp}_{cr}_{rcn}.json'
    if os.path.exists(file_name):
        with open(file_name) as json_file:
            json_results = json.load(json_file)
            histories = json_results["results"]
    else:
        histories = []
    results = []
    number_of_tries_param = number_of_tries - len(histories)
    for i in range(number_of_tries_param):
        cs_algo = ClonalSelectionAntiWorseProElite(
            problem=problem,
            population_size=ps,
            selection_size=int(ss * ps),
            clone_rate=int(cr * ps),
            random_cells_number=int(rcn * ps),
            pull_probability=pull,
            push_probability=push,
            random_probability=random,
            mutation_probability=mp / n,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            #             debug=True
        )

        cs_algo.run()
        results.append(cs_algo.get_result())
        histories.append([s.objectives[0] for s in cs_algo.history])
    if number_of_tries_param:
        print('Algorithm: ' + cs_algo.get_name())
        print('Problem: ' + problem.get_name())
        print('Solution: ' + str(get_mean_solution(results)))
        print('Fitness:  ' + str(get_mean_result(results)))
        cs_history = get_mean_history(histories)

        results = {
            "problem": problem.get_name(),
            "number_of_variables": n,
            "population_size": ps,
            "selection_size": ss,
            "pull": pull,
            "push": push,
            "random": random,
            "mutation_probability": mp / n,
            "clone_rate": cr,
            "random_cells_number": rcn,
            "results": histories}
        with open(
                f'results_elite/clonal_selection_elite_{problem.get_name()}_{n}_{ps}_{ss}_{pull}_{push}_{random}_{mp}_{cr}_{rcn}.json',
                'w') as outfile:
            json.dump(results, outfile)

# In[ ]:
