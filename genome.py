#Defines the basic genome class used for the genetic algorithms
#This is designed to function like the genomes described in the Such et al paper
#But is also general enough that all variants should be able to easily inherit from it

import gym
import torch
import torch.nn as nn
import random
import time
import multiprocessing


#Might be needed for MT
def nextStep(action, env, queue):
    observation, reward, done, _ = env.step(action)
    queue.put([(observation, reward, done), env])
    print("done")
    
"""
def retNextStep():
    def nextStep(action, env, queue):
        print("starting")
        observation, reward, done, _ = env.step(action)
        queue.put([(observation, reward, done), env])
        print("done")
    return nextStep
"""

class Genome():

    def __init__(self, experiment, randomize=True):
        self.experiment = experiment
        self.genotype = [] #Genotype will be filled with at least 2 entries, and will always be pairs of weight/bias tensors
        self.fitness = float("-inf") #default fitness is minimum before any evaluation
        self.mutate_effect = experiment.mutate_effect
        self.device = experiment.device #Device that network is evaluated on
        self.env = None
        self.species = None
        self.model = None
        #Now initialize the random genotype
        if randomize:
            self.env = experiment.env
            if experiment.layers == 0:
                self.genotype.append((torch.randn(experiment.inputs, experiment.outputs)) * (1/experiment.inputs)) #Weights
                self.genotype.append(torch.zeros(experiment.outputs)) #Bias
            else:
                self.genotype.append(torch.randn(experiment.inputs, experiment.layer_size) * (1/experiment.inputs))
                self.genotype.append(torch.zeros(experiment.layer_size))
                for i in range (experiment.layers - 1):
                    self.genotype.append(torch.randn(experiment.layer_size, experiment.layer_size) * (1/experiment.layer_size))
                    self.genotype.append(torch.zeros(experiment.layer_size))
                self.genotype.append((torch.randn(experiment.layer_size, experiment.outputs)) * (1/experiment.layer_size))
                self.genotype.append(torch.zeros(experiment.outputs))
            #self.rebuildModel()
            
    def printToTerminal(self):
        for g in self.genotype:
            print(g)

    #Makes a new neural network for the genome based on its current genotype
    def rebuildModel(self):
        model = GenomeNetwork(self.genotype, self.device, self.experiment)
        self.model = model#.to(torch.device(self.device))
    
    #Runs the environment with the network selecting actions to evaluate fitness
    def evalFitness(self, render=False, iters=1, return_frames=False):
        self.rebuildModel()
        sum_reward = 0
        trials = self.experiment.trials*iters
        frames = 0
        for _ in range(trials):
            env = self.env
            observation = env.reset()
            for t in range(20000):
                if render:
                    time.sleep(0.02)
                    env.render()
                inputs = torch.from_numpy(observation)
                inputs = (inputs.double()).to(torch.device(self.device))
                outputs = self.model(inputs)
                del inputs
                action = (torch.max(outputs, 0)[1].item())
                observation, reward, done, _ = env.step(action)
                sum_reward += reward
                frames += 1
                if done:
                    break
            env.close()
        fitness = (sum_reward/trials)
        self.model = None
        #Fitnesses of zero or less screw things up, so we fix that
        if fitness == 0:
            fitness = 0.01
        self.fitness = fitness
        if return_frames:
            return (fitness, frames)
        else:
            return fitness
    
    #This basic genome has no speciation, but the algorithm assumes there is speciation
    #So this func returns dist 0 for all genomes to create single-species behavior
    def speciesDistance(self, other):
        return 0

    #mutates based on mutation rate given by experiment
    def mutate(self):
        new = Genome(self.experiment, False) #Not creating initial random, so new.genotype is an empty list
        new.env = self.env
        for i in range(len(self.genotype)):
            new.genotype.append(self.genotype[i] + (torch.randn(self.genotype[i].size()) * (self.mutate_effect)))
        #new.rebuildModel()
        return new

#Pytorch neural net class to serve as the actual phenotype
#A new one is created each time the genotype is changed (old nets are discarded)
class GenomeNetwork(nn.Module):

    def __init__(self, genotype, device, experiment):
        super().__init__()
        self.device = device
        cuda_genes = []
        for g in genotype:
            cuda_genes.append(g.to(torch.device(self.device)))
        self.genotype = cuda_genes
        self.activation = experiment.activation_func
        self.const = experiment.activation_const

    def forward(self, inputs):
        for i in range(len(self.genotype)//2): #int division, but genotype should always be even length
            inputs = self.activation(((inputs @ self.genotype[2*i]) + self.genotype[2*i + 1])*self.const)
            #inputs.to(torch.device(self.device)) #Is this needed?
        #Neat doesn't need softmax, but other thing does
        soft = nn.Softmax(dim=0)
        return soft(inputs)
