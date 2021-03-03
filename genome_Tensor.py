import random
import copy
import torch
import math

from genome import *

#New experimental genome type; seeks to recreat NEAT with evolving tensors/layers rather than connections
#Retains many of the same ideas about crossover, mutation, speciation, etc.

class TensorNEATGenome(Genome):
    
    def __init__(self, experiment, randomize=True):
        self.experiment = experiment
        self.weights = []
        self.biases = []
        self.layer_size = 0 #Will be set after the first mutation
        self.layer_count = 0 #Tracks the number of hidden layers only
        self.fitness = 0 #Since first evaluation guarantess positive fitness, and fitness shouldn't be used prior to that
        self.device = experiment.device
        self.env = experiment.env
        self.species = None
        self.model = None
        if randomize:
            self.layer_size = math.ceil(random.randint(0, experiment.layer_step_count)*experiment.layer_step_size+experiment.inputs) #ceil since this must be an int
            self.weights.append((torch.randn(experiment.inputs, self.layer_size)) * (experiment.mutate_effect))
            self.biases.append(torch.zeros(self.layer_size))
            self.weights.append((torch.randn(self.layer_size, experiment.outputs)) * (experiment.mutate_effect))
            self.biases.append(torch.zeros(experiment.outputs))
            self.layer_count = 1
            
    def printToTerminal(self):
        print("Network has " + str(self.layer_count) + " hidden layers, with " + str(self.layer_size) + " units each.")
        print("Tensors are:")
        for i in range(len(self.weights)):
            print(self.weights[i])
            print(self.biases[i])
            
    def rebuildModel(self):
        genotype = []
        assert len(self.weights) == len(self.biases)
        if not len(self.weights) == (self.layer_count+1):
            self.printToTerminal()
            assert False
        for i in range(len(self.weights)):
            genotype.append(self.weights[i])
            genotype.append(self.biases[i])
        model = GenomeNetwork(genotype, self.device, self.experiment)
        self.model = model
        
    def speciesDistance(self, other):
        #Currently the most basic function
        if self.layer_size == 0 or self.layer_size != other.layer_size:
            return 99 #placeholder distance; properly should be positive inf
        else:
            return 0
            
    #Mutation can perturb/reset weights and biases, and also can add or remove layers
    #adding layers is done with inserting another layer_size X layer_size matrix randomly (where possible)
    #removing is done by collapsing two hidden layers, replacing them with their matrix product, to try to not change too drastically
    def mutate(self):
        exp = self.experiment
        new = TensorNEATGenome(exp, False)
        new.layer_size = self.layer_size
        new.layer_count = self.layer_count
        for i in range(len(self.weights)):
            if random.random() < exp.weight_perturb_chance:
                new.weights.append(self.weights[i] + (torch.randn(self.weights[i].size()) * (exp.mutate_effect)))
            elif random.random() < exp.weight_reset_chance: #this won't occur with the actual given probability, but I intend to start with it at zero anyways
                new.weights.append(torch.randn(self.weights[i].size()) * (exp.mutate_effect))
            else:
                new.weights.append(copy.deepcopy(self.weights[i]))
        for i in range(len(self.biases)):
            if random.random() < exp.bias_perturb_chance:
                new.biases.append(self.biases[i] + (torch.randn(self.biases[i].size()) * (exp.mutate_effect)))
            elif random.random() < exp.bias_reset_chance:
                new.biases.append(torch.zeros(self.layer_size))
            else:
                new.biases.append(copy.deepcopy(self.biases[i]))
        #New network has a full set of layers now, so we can add or remove them
        if new.layer_count < exp.max_network_size and random.random() < exp.layer_add_chance:
            assert new.layer_count > 0
            index = random.randint(1, new.layer_count)
            new.weights.insert(index, torch.eye(new.layer_size))
            new.biases.insert(index, torch.zeros(new.layer_size))
            new.layer_count += 1
        if new.layer_count > 1 and random.random() < exp.layer_collapse_chance:
            index = random.randint(0, new.layer_count-2) #can't collapse the last tensor into another one since the biases will have different sizes
            new.weights.insert(index, new.weights[index] @ new.weights[index+1])
            del new.weights[index+1] #The two being collapsed are displaced to the right, so we remove the next two
            del new.weights[index+1] #The next one has moved back, so remove the same spot again 
            new.biases.insert(index, new.biases[index] + new.biases[index+1])
            del new.biases[index+1]
            del new.biases[index+1]
            new.layer_count -= 1
        return new

    #Crossover is very simple currently since it only occurs between networks with hidden layers of the same size
    #Currently creates a genome of pretty random length
    #I can go back to my original plan of genome with length equal to the average of the two parents, but that encourages them to converge to the same length faster I think
    def crossover(self, other):
        new = TensorNEATGenome(self.experiment, False)
        left_index = random.randint(0, self.layer_count-1)
        right_index = random.randint(1, other.layer_count)
        new.layer_size = self.layer_size
        for i in range(left_index+1):
            new.weights.append(self.weights[i])
            new.biases.append(self.biases[i])
        for j in range(other.layer_count+1-right_index):
            new.weights.append(other.weights[j+right_index])
            new.biases.append(other.biases[j+right_index])
        new.layer_count = len(new.weights) - 1
        return new


















    