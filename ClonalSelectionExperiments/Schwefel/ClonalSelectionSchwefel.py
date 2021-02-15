#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys

sys.path.append("../../")

# In[7]:


from clonal_selection.clonal_selection_anti_worse_pro_elite import ClonalSelectionAntiWorseProElite
from clonal_selection.clonal_selection_cognitive import ClonalSelectionCognitive
from clonal_selection.util import get_mean_solution, get_mean_result, get_mean_history
from framework.problems.singleobjective.schwefel import Schwefel
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.operator import PolynomialMutation, SimpleRandomMutation

from clonal_selection.clonal_selection import ClonalSelection
import pandas as pd
import time

# In[14]:


import itertools
from pprint import pprint
import json

# # Clonal Selection

# In[10]:


number_of_variables = [50, 100]  # 500?
population_size = [100, 200]
selection_size = [2 / 20, 5 / 20]
mutation = ["polynomial", "random"]
mutation_probability = [1, 3]
clone_rate = [1 / 20, 2 / 20]
random_cells_number = [2 / 20, 5 / 20]

grid = [number_of_variables, population_size, selection_size, mutation, mutation_probability, clone_rate,
        random_cells_number]

grid = list(itertools.product(*grid))
# pprint(grid[:5])
print(len(grid))

# In[16]:


max_evaluations = 800
number_of_tries = 3

for n, ps, ss, m, mp, cr, rcn in grid:
    problem = Schwefel(number_of_variables=n, lower_bound=-500, upper_bound=500)
    results = []
    histories = []
    for i in range(number_of_tries):
        cs_algo = ClonalSelection(
            problem=problem,
            population_size=ps,
            selection_size=int(ss * ps),
            clone_rate=int(cr * ps),
            random_cells_number=int(rcn * ps),
            mutation=PolynomialMutation(
                probability=mp / problem.number_of_variables) if m == "polynomial" else SimpleRandomMutation(
                probability=mp / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            #             debug=True
        )

        cs_algo.run()
        results.append(cs_algo.get_result())
        histories.append([s.objectives[0] for s in cs_algo.history])

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
        "mutation": m,
        "mutation_probability": mp,
        "clone_rate": cr,
        "random_cells_number": rcn,
        "results": histories}
    with open(f'results/clonal_selection_{problem.get_name()}_{n}_{ps}_{ss}_{m}_{mp}_{cr}_{rcn}.json', 'w') as outfile:
        json.dump(results, outfile)

# In[ ]:
