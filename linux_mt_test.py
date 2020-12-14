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
import experiments
    
def multiEvalFitness(genome):
    return genome.evalFitness()

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    experiment = experiments.cart_pole
    pool = Pool(experiment.thread_count)
    #torch.multiprocessing.set_start_method('spawn')
    thread_count = experiment.thread_count
    pop_size = experiment.population
    generation_count = experiment.generations
    mutate_range = experiment.mutate_range
    population = Population()
    sys.stdout.write("Evaluating Intial Fitness:")
    sys.stdout.flush()
    thread_list = []
    new_nets = []
    for i in range(pop_size):
        new_net = Genome(experiment)
        new_net.evalFitness()
        new_nets.append(new_net)
    iters_required = math.ceil(pop_size/thread_count)
    for _ in range(iters_required):
        threads = min(thread_count, len(new_nets))
        unevaled_nets = []
        for i in range(threads):
            unevaled_nets.append(new_nets[i])
        for _ in range(threads):
            del new_nets[0] #Check for bug/change line? inefficient at best
        fitnesses = pool.map(multiEvalFitness, unevaled_nets)
        for i in range(threads):
            unevaled_nets[i].fitness = fitnesses[i]
        for net in unevaled_nets:
            population.add(net)
    print("Done!")
            
"""
def doSomething(x):
    return x*x

if __name__ == "__main__":
    threads = 8
    pool = Pool(threads)
    iterations = 101
    for _ in range(iterations):
        inputs = []
        for i in range(threads):
            inputs.append(i)
        results = pool.map(doSomething, inputs)
    print(results)
"""