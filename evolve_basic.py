#Main file for basic GA program
#Runs evolution for any genome class

import random
import copy
import sys
import time
import math
import pickle
import torch
from genome import *
from genome_NEAT import *
from population import *

#Runs basic evolution on the given experiment and params
#Creates a new generation through a combination of methods:
#Crossover from two parents, mutation from one parent, or elitism
#Ratios of this are specified by experiment
def evolve(experiment):
    total_frames = 0
    time_start = time.perf_counter()
    #Set params based on the current experiment
    pop_size = experiment.population
    generation_count = experiment.generations
    
    #Create new random population, sorted by starting fitness
    population = Population(experiment)
    new_nets = []
    saved = [] #Saving fittest from each gen to pickle file
    sys.stdout.write("Evaluating Intial Fitness:")
    sys.stdout.flush()
    for i in range(pop_size):
        new_net = "Maybe I can write a function to make a new net of type specified by the experiment"
        if experiment.genome == 'NEAT':
            new_net = NEATGenome(experiment)
        else:
            new_net = Genome(experiment)
        new_nets.append(new_net)
        
    #Fitness Eval
    for net in new_nets:
        fitness, frames = net.evalFitness(return_frames=True)
        total_frames += frames
        population.add(net)

    #Run the main algorithm over many generations
    #for g in range(generation_count):
    generation = 0
    while total_frames < experiment.max_frames:
        #First print reports on generation:
        #Debugging report I hope to remove soon
        if generation%20 == 0:
            print("Species Report: size, gens_since_improvement, record fitness, current fitnes")
            for s in population.species:
                if s.size() > 0:
                    print(s.size(), s.gens_since_improvement, s.last_fittest, s.genomes[0].fitness)
            #print(torch.cuda.memory_summary())
        gen_report_string = "\nGeneration " + str(generation) + "\nTotal frames used: " + str(total_frames) + "\nHighest Fitness: "+ str(population.fittest().fitness) + "\nTotal elapsed time:" + str(time.perf_counter() - time_start) + " seconds\n"
        sys.stdout.write(gen_report_string)
        sys.stdout.flush()
            
        #Next do the speciation work
        #Check all species and remove those that haven't improved in so many generations
        population.checkSpeciesForImprovement(experiment.gens_to_improve)
        #Adjust fitness of each individual with fitness sharing
        #Done here so it is skipped for the final population, where plain maximum fitness is desired
        if experiment.fitness_sharing:
            for species in population.species:
                for genome in species:
                    genome.fitness = genome.fitness/species.size()
        #Population is re-ordered afterwards based on new fitness
        population.reorder() #make sure this is only called when necessary
        #Assign how many offspring each species gets based on fitness of the species
        population.assignOffspringProportions()
        #Now we set the crossover/mutate counts for NEAT; since speciated evolution has a varying count of elite genomes retained
        elite_count = 0
        for species in population.species:
            if species.size() >= experiment.elite_threshold and species.gens_since_improvement < experiment.gens_to_improve:
                elite_count += experiment.elite_per_species
        experiment.elite_count = elite_count
        experiment.mutate_count = math.floor(experiment.mutate_ratio*(experiment.population - experiment.elite_count))
        experiment.crossover_count = experiment.population - experiment.mutate_count - experiment.elite_count
        #Make the new population to fill this generation
        new_pop = Population(experiment)
        #Now we select species reps for the new pop based on the old one
        for s in population.species:
            rep = population.randOfSpecies(s)
<<<<<<< HEAD
            new_species = Species(experiment, rep, False) #The genome is copied over as a rep but not added
            #rep.species = new_ispecies
=======
            new_species = Species(experiment, rep, False, s.gens_since_improvement, s.last_fittest, s.can_reproduce) #The genome is copied over as a rep but not added
>>>>>>> ec4f1f2fef0e07125edbeef3d402be04784f9cf5
            new_pop.species.append(new_species)

        time.sleep(3)            
        print("Creating new networks")

        new_nets = []  
            
        #Crossover is done first
        #Roll the dice for interspecies; limit to one per gen to make this calculation simpler
        if random.random() < experiment.interspecies_crossover*experiment.crossover_count:
            parent1 = population.select()
            parent2 = population.select()
            while parent1 == parent2:
                parent2 = population.select()
            new_net = parent1.crossover(parent2)
            new_nets.append(new_net)
            experiment.crossover_count -= 1
        #Create and add them to the pop, subtract from crossover count
        #Since we round up to the nearest integer to find how many to take from each species, the total number at the end will likely exceed the desired number
        #We fix this at the end through random pruning
        offspring = []
        for species in population.species:
            count = math.ceil(experiment.crossover_count * species.offspring_proportion)
            for _ in range(count):
                if species.size() == 1:
                    #Just do mutation
                    parent = species.select()
                    new_net = parent.mutate()
                    offspring.append(new_net)
                else: #It's possible that this has to repeat a lot if there's only a few with a large fit diff, but unlikely
                    parent1 = species.select()
                    parent2 = species.select()
                    while parent1 == parent2:
                        parent2 = species.select()
                    new_net = parent1.crossover(parent2)
                    offspring.append(new_net)
                    #Do the crossover here
        #Now remove from mutated at random until we have the right number
        to_remove = len(offspring) - experiment.crossover_count
        if to_remove < 0:
            print(to_remove)
            assert False
        for _ in range(to_remove):
            del offspring[random.randint(0, len(offspring)-1)]
        for n in offspring:
            new_nets.append(n)
                
        #Mutation to create offspring is done next; the same pruning method is used
        mutated = []
        for species in population.species:
            for _ in range(math.ceil(experiment.mutate_count * species.offspring_proportion)):
                parent = species.select()
                new_net = parent.mutate()
                mutated.append(new_net)
        #Now remove from mutated at random until we have the right number
        to_remove = len(mutated) - experiment.mutate_count
        assert (to_remove >= 0)
        for _ in range(to_remove):
            del mutated[random.randint(0, len(mutated)-1)]
        for n in mutated:
            new_nets.append(n)
        
        time.sleep(7)
        print("Evaluating new networks")
            #Fitness Eval
        for net in new_nets:
            fitness, frames = net.evalFitness(return_frames=True)
            total_frames += frames
            new_pop.add(net)
        
        print("Selecting elite networks")

        #Elite Carry-over; re-evaluates fitness first before selection
        #Currently not built to carry best of each species over; this should be handled by fitness sharing
        #And since this is typically only 1, we just want the fittest genome regardless of species
        for species in population.species:
            if species.size() >= experiment.elite_threshold and species.gens_since_improvement < experiment.gens_to_improve:
                for i in range(experiment.elite_per_species): #This needs to be redone for elite_count > 1; currently would just take best genome twice
                    best_fitness = float('-inf')
                    fittest = None
                    for i in range(experiment.elite_range):
                        fitsum = 0
                        for j in range(experiment.elite_evals):
                            fitsum += species[i].evalFitness() #eval will also return the new fitness, not just update it
                        if fitsum/experiment.elite_evals > best_fitness:
                            best_fitness = fitsum/experiment.elite_evals
                            fittest = species[i]
                    new_pop.add(fittest)
                    #Save each elite carryover to pickle file
                    save_copy = copy.deepcopy(fittest)
                    saved.append([save_copy, best_fitness])
        file = open("experiment_" + str(generation), 'wb')
        pickle.dump(experiment, file)
        population.species = []
        population.genomes = []
        population = new_pop
        generation += 1
    print("Final frame count:", str(total_frames))
    return population, saved
