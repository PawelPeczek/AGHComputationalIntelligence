# Documentation

## 1. Code

### de.py

This file contains DifferentialEvolution class. It extends the EvolutionaryAlgorithm class from JMetal. It contains the funcionality of basic differential evolution algorithm as well as the changes that were proposed throughout the course.
It accepts the following keyword arguments:
```
	problem (DEProblem) - the problem definition contained in a DEProblem class.
	each_species_size (int) - size of each species.
	max_iter (int) - it's the total number of iterations performed by an algorithm.
	cr (float) - it's the "Crossover Probability" parameter from standard Differential Evolution algorithm; should be between [0,1].
	f (float) - it's the "Differential Weight" parameter from standardDifferential Evolution algorithm; should be between [0,2].
	meeting_frequency (int) - number of iterations that's between each species meeting. The higher the number, the rarer the meetings. By default it's set to 20.
	exchange_rate (float) - rate of the population/species that will "exchange" during species meetings. By default it's set to 0.1.
	no_species (int) - number of species; the number of all points/individuals is equal to (no_species * each_species_size). By default it's set to 10.
	is_simulated_annealing (boolean) - modification that changes the crossover strategy - the specimen are selected based on radius from the base specimen, the higher the iteration number, the lower the radius. It's rather computational demanding and probably not very effective. By default it's set to False.
	number_of_partners (int) - it allows to change the standard crossover formula to a new one that allows more specimen. By default it's set to 3, and that's also the standard variant of the algorithm.
```

### de_elitism.py

It extends our DifferentialEvolution class and additionally introduces "elitism" strategy that splits the specimen into species based on their fitness.

On top of arguments in the previous point, it accepts the following argument
```
	no_partner_for_each_class (List[int]) - list of numbers of partners for each class. In general, we would prefer the best solutions to use the fewest partners and the worst solutions to use higher number of partners. By default it's set to [3, 5, 7, 9, 11].
```

## 2. Runners

Important: we had to set "PYTHONPATH" variable to a repository directory, so that the runners could be launched without any packages import issues.

### Framework runners (e.g. griewank_runner.py)

They produce single run results and allow for easy reconfiguration of an algorithm. It saves results to a CSV file containing the best specimen, the best specimen's fitness and total computation time. By default those runners have implemented number of runs with different parameters but those are easily changable to each own needs. Framework runners may seem slower than the specific runner. It does not allow a multiple runs in an easy way so we implemented our own runner.

### Differential evolution specific runner (runner.py)

This runner produces CSV file that contains average fitness (of customizable number of runs, unlike the framework runners), the best specimen and average total computational time.