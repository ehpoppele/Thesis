import basic_evolve
import evolve_multithreaded
import torch
import gym
import random
import experiments
#import time

#experiment = {"name" : 'CartPole-v0', "inputs" : 4, "outputs" : 2, "layers" : 3, "layer_size" : 16, "trials" : 20}
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    experiment = experiments.cart_multithread
    #experiment = experiments.frostbite_1
    #experiment = experiments.c_p_NEAT
    #experiment.device = 'cpu'
    #fit_pop = basic_evolve.evolve(experiment)
    fit_pop = evolve_multithreaded.evolve(experiment)
    print("fittest:", fit_pop[0].fitness)
    fittest = fit_pop[0]
    input("press enter to continue to animation")
    fittest.experiment.trials = 1
    print(fittest.evalFitness(True))
