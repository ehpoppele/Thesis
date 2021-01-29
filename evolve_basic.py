#Main file for basic GA program
#camelCaps is for func names, snake_case is variables/objects
import random
import sys
import pickle
import time
import threading
from genome import *
from genome_NEAT import *
from population import *

#Runs basic evolution on the given experiment and params
#Creates a new generation through a combination of methods:
#Crossover from two parents, mutation from one parent, or elitism
#Ratios of this are specified by exp. and currently can't apply mutation to crossover
def evolve(experiment):
    experiment.NODE_INNOVATION_NUMBER = -1
    experiment.WEIGHT_INNOVATION_NUMBER = -1
    time_start = time.perf_counter()
    #Set params based on the current experiment (so no experiment. everywhere)
    pop_size = experiment.population
    generation_count = experiment.generations
    mutate_range = experiment.mutate_range
    outfile = experiment.outfile        
    #Create new random population, sort by starting fitness
    population = Population()
    if experiment.genome == "NEAT":
        population.is_speciated = True
    saved = [] #Saving fittest from each gen to pickle file
    if outfile == 'terminal':
        sys.stdout.write("Evaluating Intial Fitness:")
        sys.stdout.flush()
    for i in range(pop_size):
        if outfile == 'terminal':
            if (10*i % pop_size < 1): #This and other prints are not working right; need to modify to be one-tenth more precisely
                sys.stdout.write(".")
                sys.stdout.flush()
        new_net = "placeholder string because isn't python funny"
        if experiment.genome == 'NEAT':
            new_net = NEATGenome(experiment)
        else:
            new_net = Genome(experiment)
        new_net.evalFitness()
        #Maybe have fitness auto-evaled when new genome is made? maybe not
        #Add new genome to the population, keeping population sorted by fitness
        #If population becomes a class, this would be moved into a method for it (pop push something etc)
        population.add(new_net)
    #Since all genomes start in the same species, we just need 1 random rep
    for g in range(generation_count):
        #print(torch.cuda.memory_summary())
        print(str(time.perf_counter() - time_start) + " elapsed seconds")
        #if outfile == 'terminal':
        sys.stdout.write("\nGeneration " +str(g) + " highest fitness: " + str(population.fittest(1).fitness) + "\n")
        sys.stdout.flush()
        if outfile != 'terminal':
            f = open(outfile, "a")
            f.write(str(g) +'\t' + str(population.fittest(1).fitness) + "\n")
            f.close()
        #Adjust fitness of each individual with fitness sharing
        #Done here so it is skipped for the final population, where plain maximum fitness is desired
        if experiment.fitness_sharing:
            for g in population:
                g.fitness = g.fitness/len(population.species_memo[g.species])
        #Check all species and remove those that haven't improved in so many generations
        #To avoid changing pop size, they aren't removed but have all fitness values set to zero
        for species in population.species:
            species.checkForImprovement(experiment.)
        #Set whole species to zero fitness if this happens (i think that's the safest way to stop them)
        #Population is re-ordered afterwards based on new fitness
        population.reorder() #make sure this is only called when necessary
        new_pop = Population()
        #Now we select species reps for the new pop based on the old one
        for species in population.species:
            rep = population
            new_species = Species(rep, False) #The genome is copied over as a rep but not added
            new_pop.species.append(new_species)
        #Crossover! With speciation checking
        sys.stdout.write("Crossover")
        sys.stdout.flush()    
        assert (experiment.crossover_range > 1) #To avoid infinite loops below; I need to update this now that Speciation checking makes this work differently
        i = 0
        set_prime = population.fitSet(experiment.crossover_range) #set of parent genomes to be used in crossover for whole procedure
        #Loop creates 1 new child genome at a time
        while i < experiment.crossover_count and len(set_prime) > 1:
            #Select two without replacement
            fit_set = population.fitSet(experiment.crossover_range)
            parent1 = random.choice(tuple(set_prime))
            fit_set.remove(parent1)
            parent2 = random.choice(tuple(fit_set)) #This can be speeded up if we don't allow it to pick things missing from set_prime; but at present it should still be correct at least
            fit_set.remove(parent2)
            #Reselect until same species or out of genomes
            while not(parent1.species == parent2.species) and bool(fit_set):
                parent2 = random.choice(tuple(fit_set))
                fit_set.remove(parent2)
            #if we haven't found a match of the same species, then we remove parent 1 from the prime set
            if (not (parent1.species == parent2.species)):
                set_prime.remove(parent1)
            else:
                new_net = parent1.crossover(parent2)
                new_net.evalFitness()
                new_pop.add(new_net)
                i += 1
        if i < experiment.crossover_count:
            print("Top individuals are all of different species and crossover is impossible. Ending the experiment early.")
            if experiment.genome_file:
                file = open(experiment.genome_file, 'wb')
                pickle.dump(saved, file)
            return population
        #Mutation second; maybe should be first?
        sys.stdout.write("Mutating")
        sys.stdout.flush()
        for i in range(experiment.mutate_count):
            if outfile == 'terminal':
                if (10*i % experiment.mutate_count < 1):
                    sys.stdout.write(".")
                    sys.stdout.flush()
            parent = population.fittest(mutate_range)
            new_net = parent.mutate()
            new_net.evalFitness()
            new_pop.add(new_net)
        #Elite Carry-over; re-evaluates fitness first before selection
        #Currently not built to carry best of each species over; this should be handled by fitness sharing
        #And since this is typically only 1, we just want the fittest genome regardless of species
        if outfile == 'terminal':
            sys.stdout.write("\nSelecting Elite")
            sys.stdout.flush()
        for i in range(experiment.elite_count): #This needs to be redone for elite_count > 1; currently would just take best genome twice
            best_fitness = float('-inf')
            fittest = None
            for i in range(experiment.elite_range):
                if outfile == 'terminal':
                    sys.stdout.write(".")
                    sys.stdout.flush()
                fitsum = 0
                for j in range(experiment.elite_evals):
                    fitsum += population[i].evalFitness() #eval will also return the new fitness, not just update it
                if fitsum/experiment.elite_evals > best_fitness:
                    best_fitness = fitsum/experiment.elite_evals
                    fittest = population[i]
            new_pop.add(fittest)
            if outfile == 'terminal':
                print("\nBest elite fitness is: ", best_fitness)
            #Save each elite carryover to list
            saved.append(fittest)
        population = new_pop
    if experiment.genome_file:
        file = open(experiment.genome_file, 'wb')
        pickle.dump(saved, file)
    return population
