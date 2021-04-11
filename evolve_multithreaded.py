#Main file for basic GA program
#Runs evolution for any genome class

import random
import copy
import sys
import time
import math
import pickle
import torch
from torch.multiprocessing import Pool, set_start_method
from genome import *
from genome_NEAT import *
from genome_Tensor import *
from population import *


"""
def multiEvalFitness(genome_list):
    torch.set_default_tensor_type(torch.DoubleTensor)
    ret = []
    frames_used = 0
    for g in genome_list:
        fitness, frames = g.evalFitness(return_frames=True)
        frames_used += frames
        ret.append(fitness)
    return (ret, frames_used)
"""
def multiEvalFitness(genome):
    torch.set_default_tensor_type(torch.DoubleTensor)
    return genome.evalFitness(return_frames=True)

def multiEvalFitnessElite(g):
    torch.set_default_tensor_type(torch.DoubleTensor)
    return (g.evalFitness(iters=g.experiment.elite_evals, return_frames=True))
    
    
def multiEvalFitnessThirty(g):
    torch.set_default_tensor_type(torch.DoubleTensor)
    return (g.evalFitness(iters=50))
    
#Runs basic evolution on the given experiment and params
#Creates a new generation through a combination of methods:
#Crossover from two parents, mutation from one parent, or elitism
#Ratios of this are specified by experiment
def evolve(experiment):
    total_frames = 0
    #If CUDA is breaking try adding this and removing it from main/run_experiment
    #try:
    #    set_start_method('spawn')
    #except RuntimeError:
    #    pass
    pool = Pool(experiment.thread_count)
    thread_count = experiment.thread_count
    time_start = time.perf_counter()
    #Set params based on the current experiment
    pop_size = experiment.population
    generation_count = experiment.generations
    
    #Create new random population, sorted by starting fitness
    population = Population(experiment)
    new_nets = []
    saved = [] #Saving fittest from each gen to pickle file
    #sys.stdout.write("Evaluating Intial Fitness:")
    #sys.stdout.flush()
    for i in range(pop_size):
        new_net = "Maybe I can write a function to make a new net of type specified by the experiment"
        if experiment.genome == 'NEAT':
            new_net = NEATGenome(experiment)
        elif experiment.genome == 'TensorNEAT':
            new_net = TensorNEATGenome(experiment)
        else:
            new_net = Genome(experiment)
        new_nets.append(new_net)
        
    #Multithreaded fitness evaluation
    """
    net_copies = []
    for _ in range(thread_count):
        net_copies.append([])
    for i in range(pop_size):
        net_copies[i%thread_count].append(copy.deepcopy(new_nets[i]))
    multiReturn = pool.map(multiEvalFitness, net_copies)
    fitnesses = []
    for thread in multiReturn:
        fitnesses.append(thread[0])
        total_frames += thread[1]
    for i in range(pop_size):
        new_nets[i].fitness = fitnesses[i%thread_count][i//thread_count]
    for net in new_nets:
        population.add(net)
    """
    #net_copies = []
    #for i in range(pop_size):
    #    net_copies.append(copy.deepcopy(new_nets[i]))
    multiReturn = pool.map(multiEvalFitness, new_nets)
    for i in range(pop_size):
        #print(new_nets[i].fitness)
        new_nets[i].fitness = multiReturn[i][0]
        #sys.stdout.write(str(multiReturn[i][0]) + "\n")
        #sys.stdout.flush()
        total_frames += multiReturn[i][1]
    for net in new_nets:
        population.add(net)

    #Run the main algorithm over many generations
    #for g in range(generation_count):
    generation = 0
    while total_frames < experiment.max_frames:
        #First print reports on generation:
        #Debugging report I hope to remove soon
        """
        if generation%20 == 0:
            print("Top genomes:")
            population[0].printToTerminal()
            population[1].printToTerminal()
            population[2].printToTerminal()
            print("Species Report: size, gens_since_improvement, record fitness, current fitness")
            for s in population.species:
                if s.size() > 0:
                    print(s.size(), s.gens_since_improvement, s.last_fittest, s.genomes[0].fitness)
            #print(torch.cuda.memory_summary())
        gen_report_string = "\nGeneration " + str(generation) + "\nTotal frames used: " + str(total_frames) + "\nHighest Fitness: "+ str(population.fittest().fitness) + "\nTotal elapsed time:" + str(time.perf_counter() - time_start) + " seconds\n"
        sys.stdout.write(gen_report_string)
        sys.stdout.flush()
        """

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
            new_species = Species(experiment, rep, False, s.gens_since_improvement, s.last_fittest, s.can_reproduce) #The genome is copied over as a rep but not added
            new_pop.species.append(new_species)

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
        
        #net_copies = []
        #for i in range(pop_size-elite_count):
        #    net_copies.append(copy.deepcopy(new_nets[i]))
        multiReturn = pool.map(multiEvalFitness, new_nets)
        for i in range(pop_size-elite_count):
            new_nets[i].fitness = multiReturn[i][0]
            total_frames += multiReturn[i][1]
        for net in new_nets:
            new_pop.add(net)

        #Elite Carry-over; re-evaluates fitness first before selection
        #Currently not built to carry best of each species over; this should be handled by fitness sharing
        #And since this is typically only 1, we just want the fittest genome regardless of species
        
        #Eval all the elite nets many times
        elite_nets = []
        for species in population.species:
            if species.size() >= experiment.elite_threshold and species.gens_since_improvement < experiment.gens_to_improve:
                for i in range(experiment.elite_range):
                    elite_nets.append(species[i])
        #net_copies = []
        #for i in range(len(elite_nets)):
        #    net_copies.append(copy.deepcopy(elite_nets[i]))
        multiReturn = pool.map(multiEvalFitnessElite, elite_nets)
        #print(fitnesses)
        for i in range(len(elite_nets)):
            elite_nets[i].fitness = multiReturn[i][0]
            total_frames += multiReturn[i][1]
           
        elite_max = float("-inf")
        top_elite = None
        for species in population.species:
            if species.size() >= experiment.elite_threshold and species.gens_since_improvement < experiment.gens_to_improve:
                for i in range(experiment.elite_per_species): #This needs to be redone for elite_count > 1; currently would just take best genome twice
                    best_fitness = float('-inf')
                    fittest = None
                    for i in range(experiment.elite_range):
                        if species[i].fitness > best_fitness:
                            best_fitness = species[i].fitness
                            fittest = species[i]
                    if best_fitness > elite_max:
                        elite_max = best_fitness
                        top_elite = fittest
                    new_pop.add(fittest)
        #If the experiment ran enough trials, we can just use the top elite. Generally this should be reserved for single-species algorithms
        if not experiment.save_elite: #If not, run our own trials, not counting frames, to find the actual best genome from this generation to save for final evaluation
            elite_nets = []
            for i in range(10):
                elite_nets.append(population[i])
            fitnesses = pool.map(multiEvalFitnessThirty, elite_nets)
            for i in range(10):
                elite_nets[i].fitness = fitnesses[i]
               
            elite_max = float("-inf")
            top_elite = None
            for net in elite_nets:
                if net.fitness > elite_max:
                    elite_max = net.fitness
                    top_elite = net
                    
        save_copy = top_elite.newCopy() #Still need to modify/add this for regular/tensor genomes
        saved.append([save_copy, elite_max])
            
        """
        if top_elite is None:
            sys.stdout.write("No elite could be found, using pop.fittest instead\n")
            for s in population.species:
                sys.stdout.write(str(species.size()) + str(species.gens_since_improvement) + str(species.can_reproduce))
            top_elite = population.fittest()
            top_elite.evalFitness(iters=top_elite.experiment.elite_evals)#These frames are not counted since they are only for reporting purposes and do not affect the actual algorithm
            elite_max = top_elite.fitness
        if elite_max < 5.0:
            for s in population.species:
                print(s.can_reproduce)
            sys.stdout.write("Strange elite behavior. Fitness is " + str(top_elite.fitness) + " Trial is " + str(top_elite.evalFitness()) + " Top fitness is: " + str(population.fittest().fitness) + "\n")
            sys.stdout.flush()
        #Save top elite carryover to pickle file
        save_copy = copy.deepcopy(top_elite)
        saved.append([save_copy, elite_max])
        """
        total_layers = 0
        for genome in new_pop:
            total_layers += genome.layer_count
        avg_layers = total_layers/new_pop.size()
        elapsed = int(time.perf_counter()-time_start)
        time_string = str(elapsed//3600) + ":" + str((elapsed%3600)//60) + ":" + str(elapsed%60)
        sys.stdout.write(str(100*total_frames/experiment.max_frames) + "% complete | " + time_string + " elapsed | " + str(elite_max) + " recent score | " + str(save_copy.layer_size) + " " + str(save_copy.layer_count) + " layer size/count | " + str(len(new_pop.species)) + " total species | " + str(avg_layers) + " average layers\n")
        sys.stdout.flush()
        population.species = []
        population.genomes = []
        population = new_pop
        generation += 1
    print("Final frame count:", str(total_frames))
    print("Total generations:", generation)
    print("Time Elapsed:", time.perf_counter()-time_start)
    return population, saved
