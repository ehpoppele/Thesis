import evolve_basic
import evolve_multithreaded
import torch
import gym
import random
import experiments
import sys
#import torch.multiprocessing

#import time


#experiment = {"name" : 'CartPole-v0', "inputs" : 4, "outputs" : 2, "layers" : 3, "layer_size" : 16, "trials" : 20}
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    #experiment = experiments.cart_multithread
    #experiment = experiments.frostbite_1
    experiment = experiments.cart_pole
    #experiment = experiments.frost_NEAT
    #experiment.device = 'cpu'
    #fit_pop = basic_evolve.evolve(experiment)
    #fit_pop = basic_evolve.evolve(experiment)
    if len(sys.argv) > 1:
        found = False
        for e in experiments.list:
            if e.name == sys.argv[1]:
                experiment = e
                found = True
        if not found:
            print("Requested experiment not found")
            print("Running default instead")
    #fit_pop = evolve_multithreaded.evolve(experiment)
    fit_pop = evolve_basic.evolve(experiment)
    print("fittest:", fit_pop.top_fittest())
    fittest = fit_pop.top_fittest()
    input("press enter to continue to animation")
    fittest.experiment.trials = 1
    print(fittest.evalFitness(True))
    for s in fit_pop.species:
        print("fittest genome of the next species:")
        if s.size() > 0:
            print(s[0].fitness)
            s[0].printToTerminal()
        else:
            print("Species seems to be empty")
        print()
    fittest.printToTerminal()
    
