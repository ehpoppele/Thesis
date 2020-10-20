import gym
import torch
import torch.nn as nn
import random

from genome import *

#Class representing the nodes in a NEAT network
#Go into the first part of the genotype
class NEATNode():

    def __init__(self, type, layer, mutable=True):
        self.type = type #Should be string; 'input', 'output', or 'hidden'
        self.layer = layer

class NEATGenome(Genome):

    def __init__(self):
        
        
    def retraceLayers(self):
        for node in self.genotype[0]:
            
        
    def rebuildModel(self):
        self.retraceLayers()
        working_nodes = self.addShadowNodes()
        working_weights = self.addShadowWeights()
        tensor_list = []
        while not done:
            new_tensor = prev_layer x curr_layer (zeroes)
            for node in prev_layer:
                for weight in weights[prev_layer]:
                    if weights.from == node:
                        new_tensor[node.position][weight.to.position] = weight.weight
            
        
    