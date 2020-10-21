#Main file for basic GA program
#camelCaps is for func names, snake_case is variables/objects
import random
import sys
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
    sys.stdout.write("Evaluating Intial Fitness:")
    sys.stdout.flush()
    for i in range(pop_size):
        if (10*i % pop_size < 1):
            sys.stdout.write(".")
            sys.stdout.flush()
        new_net = Genome(experiment)
        new_net.evalFitness()
        #Maybe have fitness auto-evaled when new genome is made? maybe not
        #Add new genome to the population, keeping population sorted by fitness
        #If population becomes a class, this would be moved into a method for it (pop push something etc)
        population.add(new_net)
    for g in range(generation_count):
        #Print with sys so we can see output while running (sometimes won't show otherwise)
        sys.stdout.write("\nGeneration " +str(g) + " highest fitness: " + str(population.fittest(1).fitness) + "\n")
        sys.stdout.flush()
        new_pop = Population(pop_size)
        #Crossover would go right here
        #for now I only have mutation
        sys.stdout.write("Mutating:")
        sys.stdout.flush()
        for i in range(experiment.mutate_count):
            if (10*i % experiment.mutate_count < 1):
                sys.stdout.write(".")
                sys.stdout.flush()
            parent = population.fittest(mutate_range)
            new_net = parent.mutate()
            new_net.evalFitness()
            new_pop.add(new_net)
        #Elite Crossover; re-evaluates fitness first before selection
        sys.stdout.write("\nSelecting Elite")
        sys.stdout.flush()
        for i in range(experiment.elite_count):
            sys.stdout.write(".")
            sys.stdout.flush()
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
