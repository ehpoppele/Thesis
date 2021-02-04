#Another basic test file for genome functionality
#This is not a unit test nor should its ability or inability to run be taken as a sign that the genome class is actually working
#Not any longer, anyways

import population
import gym
import experiments
import sys
import pickle
import torch

#experiment = {"name" : 'CartPole-v0', "inputs" : 4, "outputs" : 2, "layers" : 3, "layer_size" : 16, "trials" : 20}
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    file = ""
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = input("Enter the name of the file to be opened")
    pop = population.Population()
    gene_list = pickle.load(open(file, "rb"))   
    for gene in gene_list:
        pop.add(gene)
    fittest = pop[0]
    print(fittest.evalFitness(True))