import gym
import torch
import torch.nn as nn
import random

class Genome():
    
    def __init__(self, experiment, randomize=True):
        self.experiment = experiment
        self.genotype = [] #Genotype will be filled with at least 2 entries, and will always be pairs of weight/bias tensors
        self.fitness = float("-inf") #default fitness is minimum before any evaluation
        self.mutate_effect = experiment.mutate_effect
        self.model = None
        #Now initialize the random genotype
        if randomize:
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
            self.rebuildModel()
            
    def rebuildModel(self):
        """"
        #There is certainly a faster way to do this, that doesn't require a python loop. I'll look into it later
        def model(inputs):
            for i in range(len(self.genotype)//2): #int division, but genotype should always be even length
                relu = nn.ReLU()
                inputs = relu((inputs @ self.genotype[2*i]) + self.genotype[2*i + 1])
            soft = nn.Softmax(dim=0)
            return soft(inputs)
        """
        model = Genome_network(self.genotype)
        self.model = model
        
    def evalFitness(self):
        sum_reward = 0
        trials = self.experiment.trials
        for i in range(trials):
            env = gym.make(self.experiment.name)
            observation = env.reset()
            for t in range(1000): #Can work with changing this to higher?
                inputs = torch.from_numpy(observation)
                inputs = inputs.double()
                outputs = self.model(inputs)
                action = 0
                rand_select = random.random()
                #Select an index i based on the output array distribution
                for i in range(len(outputs)):
                    rand_select -= outputs[i]
                    if rand_select < 0:
                        action = i
                        break
                observation, reward, done, info = env.step(action)
                sum_reward += reward
                if done:
                    break
            env.close()
        self.fitness = sum_reward/trials
        return sum_reward
        
    #mutates based on mutation rate given by experiment
    def mutate(self):
        new = Genome(self.experiment, False) #Not creating initial random, so new.genotype is an empty list
        for i in range(len(self.genotype)):
            new.genotype.append(self.genotype[i] + (torch.randn(self.genotype[i].size()) * (self.mutate_effect/self.genotype[i].size()[0])))
        new.rebuildModel()
        return new
        
class Genome_network(nn.Module):
    
    def __init__(self, genotype):
        super().__init__()
        self.genotype = genotype

    def forward(self, inputs):
        for i in range(len(self.genotype)//2): #int division, but genotype should always be even length
            relu = nn.ReLU()
            inputs = relu((inputs @ self.genotype[2*i]) + self.genotype[2*i + 1])
        soft = nn.Softmax(dim=0)
        return soft(inputs)
        
        