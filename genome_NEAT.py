import gym
import torch
import torch.nn as nn
import random

from genome import *

#Class representing the nodes in a NEAT network
#Go into the first part of the genotype
class NEATNode():

    def __init__(self, id, type, mutable=True):
        self.id = id
        self.type = type #Should be string; 'input', 'output', or 'hidden'
        

class NEATGenome(Genome):

    #Similar to regular init, but has 2 parts to genotype
    def __init__(self, experiment, randomize=True):
        self.experiment = experiment
        self.fitness = float("-inf") 
        self.mutate_effect = experiment.mutate_effect
        self.model = None
        self.device = experiment.device
        self.env = None 
        self.layers = 2 #number of layers; I should rename this
        self.nodes = []
        self.weights = []
        if randomize:
            self.env = gym.make(self.experiment.name, frameskip=4)
            #create appropriate number of nodes, then weights (review paper as to how many intial connections)
            self.rebuildModel()
        
    
    #Goes through the genotype and updates the layers/depth value of each node
    #Assumes input nodes are fixed as the only nodes at zero
    #This is very inefficient...
    def retraceLayers(self):
        layer = 0
        max_layer = 0
        while True:
            nodes_found = 0
            for node in self.nodes:
                if node.layer == layer:
                    nodes_found += 1
                    for weight in self.weights:
                        if weight.from.layer == layer:
                            weight.to.layer = layer + 1
                if node.layer > max_layer:
                    max_layer = node.layer
            if nodes_found == 0:
                break
        self.layers = max_layer
        
    #Returns a altered genotype that includes the extra nodes necessary to make complete connections
    #For the nodes and weights to be converted effectively into tensors, we need connections to stop at each layer
    #Rather than have one running through from layer 0 to 3, for example. This adds extra connections at a weight of 1
    #up to the layer before the destination layer, then alters the original weight to go just between those last two layers
    #Also, returns nodes and weights in 2D lists, separated by layer (weights attached to destination layer)
    #And returns a list with number of nodes in each layer?
    def buildShadowElements(self):
        nodes = [[]] * len(self.layers)
        for node in self.nodes:
            nodes[node.layer].append(node)
        shadowCount = 0 #Counter to tracker id for shadow node
        weights = [[]] * len(self.layers)
        for weight in self.weights:
            prev = weight.from
            for i in range(weight.to.layer - weight.from.layer - 1):
                shadowNode = NEATNode((0-shadowCount), 'hidden', weight.from + 1 + i)
                shadowCount += 1
                shadowWeight = NEATWeight(prev, shadowNode, 1.0)
                nodes[shadowNode.layer].append(shadowNode)
                weights[shadowNode.layer].append(shadowWeight)
                prev = shadowNode
            weight.from = prev 
            weights[weight.to.layer].append(weight)
        layer_counts = []
        for layer in nodes:
            layer_counts.append(len(len(layer))
        return nodes, weights, layer_counts
        
    def rebuildModel(self):
        self.retraceLayers()
        working_nodes, working_weights, layer_counts = self.buildShadowElements()
        tensor_list = []
        prev_layer = 0
        curr_layer = 1
        while not done:
            weight_tensor = torch.zeros(layer_counts[curr_layer-1], layer_counts[curr_layer])
            bias_tensor = torch.zeros(layer_counts[curr_layer])
            for i in range(layer_counts[curr_layer]):
                node = working_nodes[curr_layer][i]
                for j in range(len(working_weights[curr_layer])):
                    weight = working_weights[curr_layer][j]
                    if weights.to == node:
                        weight_tensor[i][j] = weight.value
                bias_tensor[i] = node.bias
            tensor_list.append(weight_tensor)
            tensor_list.append(bias_tensor)
            prev_layer += 1
            curr_layer += 1
            if curr_layer >= len(self.nodes):
                break
        model = Genome_network(tensor_list, self.device)
        self.model = model.to(torch.device(self.device))
        
        #Also placeholder until I write this
        def crossover(self, other):
            return self.mutate()
        
        #Placeholder mutate so I can test other things first
        def mutate(self):
            new = Genome(self.experiment)
            return new
            
        
        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
            
        
    