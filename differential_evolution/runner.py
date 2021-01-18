import os
import json
from datetime import datetime

import numpy as np

from differential_evolution.de import DifferentialEvolution
from framework.problems.singleobjective.griewank import Griewank
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from framework.problems.singleobjective.schwefel import Schwefel


def calcuate_average_fitness_per_iteration(repeats_best_fitness_per_iteration):
  average_fitness_per_iteration = list()
  
  for i in range(len(repeats_best_fitness_per_iteration[0])):
    fitness_sum = 0
    for j in range(len(repeats_best_fitness_per_iteration)):
      fitness_sum += repeats_best_fitness_per_iteration[j][i]
    fitness_average = fitness_sum / len(repeats_best_fitness_per_iteration)
    average_fitness_per_iteration.append(fitness_average)

  return average_fitness_per_iteration


def save_to_json(file_name, repeats_best_speciman, average_fitness_per_iteration, average_total_computing_time):
  print(f'Saving results to a file {file_name}...')

  data = {}

  data['repeats_best_speciman'] = repeats_best_speciman
  data['average_fitness'] = average_fitness_per_iteration
  data['average_total_comupting_time'] = average_total_computing_time

  with open(file_name, 'w') as outfile:
    json.dump(data, outfile)


def run(problem, save_file, no_iters, no_repeats, no_partners):
  no_species = 10
  cr = 0.9
  f = 0.8
  each_species_size = 50

  de = DifferentialEvolution(
    problem=problem,
    max_iter=no_iters,
    cr=cr,
    f=f,
    no_species=no_species,
    each_species_size=each_species_size,
    number_of_partners=no_partners,
    is_simulated_annealing=False
  )

  repeats_best_speciman = list()
  repeats_best_fitness_per_iteration = list()
  repeats_total_computing_time = list()

  print(f'Starting solving a {problem.get_name()} problem...')

  for i in range(no_repeats):
    print(f'Starting a repeat number {i + 1}...')

    de.run()
    result = de.get_result()

    repeats_best_speciman.append(result.variables)

    fitnesses = de.best_fitness_per_iteration
    repeats_best_fitness_per_iteration.append(fitnesses)

    repeats_total_computing_time.append(de.total_computing_time)

  average_fitness_per_iteration = calcuate_average_fitness_per_iteration(repeats_best_fitness_per_iteration)
  average_total_computing_time = np.average(np.array(repeats_total_computing_time))

  save_to_json(
    file_name=save_file,
    repeats_best_speciman=repeats_best_speciman,
    average_fitness_per_iteration=average_fitness_per_iteration,
    average_total_computing_time=average_total_computing_time
  )


if __name__ == '__main__':
  '''
  griewank_problem = Griewank(number_of_variables=10, lower_bound=-100, upper_bound=100)
  run(
    problem=griewank_problem,
    save_file='griewank_test.json',
    no_iters=100,
    no_repeats=5,
    no_partners=5
  )
  '''

  griewank_problem = Griewank(number_of_variables=10, lower_bound=-100, upper_bound=100)
  run(
    problem=griewank_problem,
    save_file='griewank_3_partners.json',
    no_iters=1000,
    no_repeats=10,
    no_partners=3
  )
  run(
    problem=griewank_problem,
    save_file='griewank_5_partners.json',
    no_iters=1000,
    no_repeats=10,
    no_partners=5
  )
  run(
    problem=griewank_problem,
    save_file='griewank_10_partners.json',
    no_iters=1000,
    no_repeats=10,
    no_partners=10
  )

  rastrigin_problem = Rastrigin(number_of_variables=10)
  run(
    problem=rastrigin_problem,
    save_file='rastrigin_3_partners.json',
    no_iters=1000,
    no_repeats=10,
    no_partners=3
  )
  run(
    problem=rastrigin_problem,
    save_file='rastrigin_5_partners.json',
    no_iters=1000,
    no_repeats=10,
    no_partners=5
  )
  run(
    problem=rastrigin_problem,
    save_file='rastrigin_10_partners.json',
    no_iters=1000,
    no_repeats=10,
    no_partners=10
  )

  schwefel_problem = Schwefel(number_of_variables=10, lower_bound=-500, upper_bound=500)
  run(
    problem=schwefel_problem,
    save_file='schwefel_3_partners.json',
    no_iters=1000,
    no_repeats=10,
    no_partners=3
  )
  run(
    problem=schwefel_problem,
    save_file='schwefel_5_partners.json',
    no_iters=1000,
    no_repeats=10,
    no_partners=5
  )
  run(
    problem=schwefel_problem,
    save_file='schwefel_10_partners.json',
    no_iters=1000,
    no_repeats=10,
    no_partners=10
  )