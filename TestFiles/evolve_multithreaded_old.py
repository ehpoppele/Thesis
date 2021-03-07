#Main file for basic GA program
#camelCaps is for func names, snake_case is variables/objects
import random
import math
import sys
import pickle
import time
from torch.multiprocessing import Pool
import gym
from genome import *
from genome_NEAT import *
from population import *

#-----------------------
#Thread helper functions
#-----------------------
 
def multiEvalFitness(genome):
    return genome.evalFitness()


#Runs basic evolution on the given experiment and params
#Creates a new generation through a combination of methods:
#Crossover from two parents, mutation from one parent, or elitism
#Ratios of this are specified by exp. and currently can't apply mutation to crossover
def evolve(experiment):
    pool = Pool(experiment.thread_count)
    #torch.multiprocessing.set_start_method('spawn')
    thread_count = experiment.thread_count
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
    saved = [] #Saving fittest from each gen to pickle file
    if outfile == 'terminal':
        sys.stdout.write("Evaluating Intial Fitness:")
        sys.stdout.flush()
    new_nets = []
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
        new_nets.append(new_net)
    iters_required = math.ceil(pop_size/thread_count)
    for _ in range(iters_required):
        threads = min(thread_count, len(new_nets))#Number of threads for this iteration; should be thread_count for all but the last, where it can be less
        net_batch = [] #The nets we are evaluating this loop
        batch_copy = [] #Copies of those nets that will get sent to the pool
        #We run into "too many files" errors if the same nets get used, but using and discarding a deepcopy seems to fix
        for i in range(threads):
            net_batch.append(new_nets[i])
            batch_copy.append(copy.deepcopy(new_nets[i]))
        for _ in range(threads):
            del new_nets[0] #Check for bug/change line? inefficient at best
        fitnesses = pool.map(multiEvalFitness, batch_copy)
        for i in range(threads):
            net_batch[i].fitness = fitnesses[i]
        for net in net_batch:
            population.add(net)
    assert len(new_nets) == 0
    print(population.size())
    species_reps = [population[0]] #List of the representatives for each species
    #assert False
    for g in range(generation_count):
        new_pop = Population()
        #print(torch.cuda.memory_summary())
        print(str(time.perf_counter() - time_start) + " elapsed seconds")
        #if outfile == 'terminal':
        sys.stdout.write("\nGeneration " +str(g) + " highest fitness: " + str(population.fittest(1).fitness) + "\n")
        sys.stdout.flush()
        if outfile != 'terminal':
            f = open(outfile, "a")
            f.write(str(g) +'\t' + str(population.fittest(1).fitness) + "\n")
            f.close()
        #new_pop = Population()
        new_nets = []
        #Crossover! With speciation checking
        sys.stdout.write("Crossover")
        sys.stdout.flush()
        population.species_memo = [] #resets the species memo before species are reassigned
        #Assign species to all genomes
        # Here #
        if (experiment.crossover_count != 0):
            assert (experiment.crossover_range > 1) #To avoid infinite loops below; I need to update this now that Speciation checking makes this work differently
        i = 0
        set_prime = population.fitSet(experiment.crossover_range)
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
            #if out of 
            if not (parent1.species == parent2.species):
                set_prime.remove(parent1)
            else:
                new_net = parent1.crossover(parent2)
                new_nets.append(new_net)
                #new_net.evalFitness()
                #new_pop.add(new_net)
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
            new_nets.append(new_net)
            #new_net.evalFitness()
            #new_pop.add(new_net)
        #Elite Carry-over; re-evaluates fitness first before selection
        sys.stdout.write("Evaluating Fitness")
        sys.stdout.flush()
        iters_required = math.ceil((pop_size-experiment.elite_count)/thread_count)
        for _ in range(iters_required):
            threads = min(thread_count, len(new_nets))#Number of threads for this iteration; should be thread_count for all but the last, where it can be less
            net_batch = [] #The nets we are evaluating this loop
            batch_copy = [] #Copies of those nets that will get sent to the pool
            #We run into "too many files" errors if the same nets get used, but using and discarding a deepcopy seems to fix
            for i in range(threads):
                net_batch.append(new_nets[i])
                batch_copy.append(copy.deepcopy(new_nets[i]))
            for _ in range(threads):
                del new_nets[0] #Check for bug/change line? inefficient at best
            fitnesses = pool.map(multiEvalFitness, batch_copy)
            for i in range(threads):
                net_batch[i].fitness = fitnesses[i]
            for net in net_batch:
                new_pop.add(net)
        assert len(new_nets) == 0
        if outfile == 'terminal':
            sys.stdout.write("\nSelecting Elite")
            sys.stdout.flush()
        unevaled_nets = []
        for i in range(experiment.elite_range):
            batch_copy.append(copy.deepcopy(population[i]))
        fitnesses = pool.map(multiEvalFitness, batch_copy)
        for i in range(threads):
            population[i].fitness = fitnesses[i]
        best_fitness = float('-inf')
        for i in range(experiment.elite_count): #This needs to be redone for elite_count > 1; currently would just take best genome twice
            fittest = None
            for i in range(experiment.elite_range):
                if outfile == 'terminal':
                    sys.stdout.write(".")
                    sys.stdout.flush()
                if population[i].fitness > best_fitness:
                    best_fitness = population[i].fitness
                    fittest = population[i]
            new_nets.append(fittest)
            if outfile == 'terminal':
                print("\nBest elite fitness is: ", best_fitness)
            #Save each elite carryover to list
            saved.append(fittest)
        for net in new_nets:
            new_pop.add(net)
        print(new_pop.size())
        assert(new_pop.size() == pop_size)
        for i in len(species_reps):
            species_reps[i] = population.randOfSpecies(i)
        population = new_pop
    if experiment.genome_file:
        file = open(experiment.genome_file, 'wb')
        pickle.dump(saved, file)
    return population
