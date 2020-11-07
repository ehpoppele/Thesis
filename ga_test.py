import basic_evolve
import torch
import gym
import random
import experiments
import time

#experiment = {"name" : 'CartPole-v0', "inputs" : 4, "outputs" : 2, "layers" : 3, "layer_size" : 16, "trials" : 20}
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    #experiment = experiments.cart_pole
    experiment = experiments.frostbite_1
    #experiment = experiments.c_p_NEAT
    #experiment.device = 'cpu'
    print(torch.cuda.memory_summary())
    fit_pop = basic_evolve.evolve(experiment)
    print(torch.cuda.memory_summary())
    print("fittest:", fit_pop[0].fitness)
    fittest = fit_pop[0]
    input("press enter to continue to animation")
    fittest.experiment.trials = 1
    print(fittest.evalFitness(True))
    copy_fit = []
    fittest = None
    for i in range(101):
        fit_pop[i].model = None
        fit_pop[i].rebuildModel()
        #fit_pop[i].experiment = None
        #fit_pop[i].genotype = None
        #fit_pop[i].fitness = None
        #fit_pop[i].mutate_effect = None
        #fit_pop[i].device = None
        #fit_pop[i].env = None
    print(torch.cuda.memory_summary())
