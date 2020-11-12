#Main file for basic GA program
#camelCaps is for func names, snake_case is variables/objects
import random
import sys
import pickle
import time
import multiprocessing
import gym
from genome import *
from genome_NEAT import *
from population import *

#-----------------------
#Thread helper functions
#-----------------------

#Multithread Initialization Function
#Returns a threadsafe func to continuosly create new genomes,
#eval them, and add them to the population
def threadCreate(population, experiment, env, queue):
    done = False
    while not done:
        new_net = "placeholder string because isn't python funny" #this is too hacky
        if experiment.genome == 'NEAT':
            new_net = NEATGenome(experiment)
        else:
            new_net = Genome(experiment)
        new_net.env = env
        new_net.evalFitness()
        if queue.full():
            done = True
        else:
            queue.put(new_net)
            sys.stdout.write("Added one new net")
            sys.stdout.flush()
    sys.stdout.write("\n Done with thread")
    sys.stdout.flush()
    """
    #get lock
    population.lock.acquire()
    if population.size() < experiment.population:
        population.add(new_net)
    else:
        done = True
    population.lock.release()
    """
    
#Returns a mutate function with fixed params for the populations
#So that threads can run them without using args
def getMutateFunc(new_pop, old_pop, experiment, env):
    def threadMutate():
        done = False
        while not done:
            parent = old_pop.fittest(experiment.mutate_range)
            new_net = parent.mutate()
            new_net.env = env
            new_net.evalFitness()
            #get lock
            new_pop.lock.acquire()
            if new_pop.size() < experiment.mutate_count:
                new_pop.add(new_net)
                print("adding")
            else:
                done = True
            new_pop.lock.release()
    return threadMutate
    
#Returns a func to run elite fitness evaluation on a single genome
#Sets that genome's fitness after wards to the result
def getEliteFunc(genome, env):
    def threadEliteEval():
        experiment = genome.experiment
        fitsum = 0
        genome.env = env
        for i in range(experiment.elite_evals):
            fitsum += genome.evalFitness()
        genome.fitness = fitsum/experiment.elite_evals
    return threadEliteEval


#Runs basic evolution on the given experiment and params
#Creates a new generation through a combination of methods:
#Crossover from two parents, mutation from one parent, or elitism
#Ratios of this are specified by exp. and currently can't apply mutation to crossover
def evolve(experiment):
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
    thread_list = [] #now just a single list for threads, emptied and reused for create/mutate/crossover/elite
    new_nets = multiprocessing.Queue(maxsize=pop_size)
    for _ in range(experiment.thread_count):
        new_env = gym.make(experiment.name)
        #thread_create_func = getCreateFunc(population, experiment, new_env)
        new_thread = multiprocessing.Process(target=threadCreate, args=(population, experiment, new_env, new_nets))
        thread_list.append(new_thread)
        new_thread.start()
    for thread in thread_list:
        print("about to join")
        thread.join()
        print(new_nets.empty())
    while not new_nets.empty():
        new_net = new_nets.get()
        population.add(new_net)
    print(population.size())
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
        new_pop = Population()
        #Crossover would go right here
        #for now I only have mutation
        sys.stdout.write("Mutating")
        sys.stdout.flush()
        thread_list = []
        for _ in range(experiment.thread_count):
            new_env = gym.make(experiment.name)
            thread_mutate_func = getMutateFunc(new_pop, population, experiment, new_env)
            new_thread = multiprocessing.Process(target=thread_mutate_func)
            thread_list.append(new_thread)
            new_thread.start()
        for thread in thread_list:
            thread.join()
        #Elite Crossover; re-evaluates fitness first before selection
        if outfile == 'terminal':
            sys.stdout.write("\nSelecting Elite")
            sys.stdout.flush()
        for i in range(experiment.elite_count): #This needs to be redone for elite_count > 1; currently would just take best genome twice
            #Different approach for threads on elite; not sure if this is bad but it's at least inconsistent with the above...
            #Since elite range should be close to/less than the number of threads, we just make that many threads, one for each genome
            #Some threads may run on the same core, but c'est la vie
            thread_list = []
            for i in range(experiment.elite_range):
                new_env = gym.make(experiment.name)
                thread_elite_func = getEliteFunc(population[i], new_env)
                new_thread = multiprocessing.Process(target=thread_elite_func)
                thread_list.append(new_thread)
                new_thread.start()
            for thread in thread_list:
                thread.join()
            best_fitness = float('-inf')
            fittest = None
            for i in range(experiment.elite_range):
                if  population[i].fitness > best_fitness:
                    best_fitness = population[i].fitness
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