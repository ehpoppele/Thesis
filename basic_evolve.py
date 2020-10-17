#Main file for basic GA program
#camelCaps is for func names, snake_case is variables/objects
import random
from genome import *
from population import *

#Runs basic evolution on the given experiment and params
#Creates a new generation through a combination of methods:
#Crossover from two parents, mutation from one parent, or elitism
#Ratios of this are specified by exp. and currently can't apply mutation to crossover
def evolve(experiment):
    #Set params based on the current experiment (so no experiment. everywhere)
    pop_size = experiment.population
    generation_count = experiment.generations
    mutate_range = experiment.mutate_range
    #Create new random population, sort by starting fitness
    population = Population(pop_size)
    for i in range(pop_size):
        new_net = Genome(experiment)
        new_net.evalFitness()
        #Maybe have fitness auto-evaled when new genome is made? maybe not
        #Add new genome to the population, keeping population sorted by fitness
        #If population becomes a class, this would be moved into a method for it (pop push something etc)
        population.add(new_net)
    for g in range(generation_count):
        print(population.fittest(1).fitness)
        new_pop = Population(pop_size)
        #Crossover would go right here
        #for now I only have mutation
        for i in range(experiment.mutate_count):
            parent = population.fittest(mutate_range)
            new_net = parent.mutate()
            new_net.evalFitness()
            new_pop.add(new_net)
        #Elite Crossover; re-evaluates fitness first before selection
        for i in range(experiment.elite_count):
            best_fitness = float('-inf')
            fittest = None
            for i in range(experiment.elite_range):
                fitsum = 0
                for j in range(experiment.elite_evals):
                    fitsum += population[i].evalFitness() #eval will also return the new fitness, not just update it
                if fitsum/experiment.elite_evals > best_fitness:
                    best_fitness = fitsum/experiment.elite_evals
                    fittest = population[i]
            new_pop.add(fittest)
        population = new_pop
    return population
                        