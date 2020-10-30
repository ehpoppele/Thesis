import gym
import torch
import torch.nn as nn
import random

from genome import *

#Class representing the nodes in a NEAT network
#Go into the first part of the genotype
class NEATNode():

    def __init__(self, id, type, layer, bias=0):
        self.id = id
        self.type = type #Should be string; 'input', 'output', or 'hidden' or 'shadow'
        self.layer = layer
        self.bias = bias
        self.l_index = 0 #Layer index, used only for rebuilding model faster
        
class NEATWeight():

    def __init__(self, origin, to, value):
        self.origin = origin #I hope this is a shallow copy
        self.to = to
        self.value = value
        self.origin_id = origin.id #Consistency across networks with same architecture but diff node objects
        self.to_id = to.id
        self.origin_copy = origin #Hold onto for when the origin is changed to a shadow node
        

class NEATGenome(Genome):

    #Similar to regular init, but has 2 parts to genotype
    def __init__(self, experiment, randomize=True):
        self.experiment = experiment
        self.fitness = float("-inf") 
        self.mutate_effect = experiment.mutate_effect
        self.m_weight_chance = experiment.mutate_odds[0]
        self.m_node_chance = experiment.mutate_odds[1]
        self.m_connection_chance = experiment.mutate_odds[2]
        self.model = None
        self.device = experiment.device
        self.env = None 
        self.layers = 2 #number of layers; I should rename this
        self.nodes = []
        self.weights = []
        if randomize:
            self.env = experiment.env
            inputs = []
            outputs = []
            for i in range(experiment.inputs):
                new_node = NEATNode(i, 'input', 0, 0)
                inputs.append(new_node)
            for i in range(experiment.outputs):
                new_node = NEATNode(i, 'output', 1, random.random()*self.experiment.inputs)
                outputs.append(new_node)
            for i in inputs:
                for o in outputs:
                    new_weight = NEATWeight(i, o, random.random()*experiment.inputs)
                    self.weights.append(new_weight)
            self.nodes += inputs
            self.nodes += outputs
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
                if node.type == 'input' and node.layer != 0:
                    for w in self.weights:
                        if w.to.type == 'input':
                            print("Input as a destination!")
                    assert False
                if node.layer == layer:
                    nodes_found += 1
                    for weight in self.weights:
                        if weight.origin.layer == layer:
                            weight.to.layer = layer + 1
                if node.layer > max_layer:
                    max_layer = node.layer
            if nodes_found == 0:
                break
            layer += 1
        self.layers = max_layer + 1
        
    #Returns a altered genotype that includes the extra nodes necessary to make complete connections
    #For the nodes and weights to be converted effectively into tensors, we need connections to stop at each layer
    #Rather than have one running through origin layer 0 to 3, for example. This adds extra connections at a weight of 1
    #up to the layer before the destination layer, then alters the original weight to go just between those last two layers
    #Also, returns nodes and weights in 2D lists, separated by layer (weights attached to destination layer)
    #And returns a list with number of nodes in each layer?
    def buildShadowElements(self):
        nodes = []
        for _ in range(self.layers):
            nodes.append([])
        for node in self.nodes:
            nodes[node.layer].append(node)
        shadowCount = 0 #Counter to tracker id for shadow node
        #weights = [[]] * self.layers cursed!
        weights = []
        for _ in range(self.layers):
            weights.append([])
        for weight in self.weights:
            prev = weight.origin
            for i in range(weight.to.layer - weight.origin.layer - 1):
                shadowNode = NEATNode((0-shadowCount), 'shadow', weight.origin.layer + 1 + i)
                shadowCount += 1
                shadowWeight = NEATWeight(prev, shadowNode, 1.0)
                nodes[shadowNode.layer].append(shadowNode)
                weights[shadowNode.layer].append(shadowWeight)
                prev = shadowNode
            weight.origin = prev #!!! Need to fix this, since it needs a origin pointer and a origin id value
            weights[weight.to.layer].append(weight)
        layer_counts = []
        for layer in nodes:
            layer_counts.append(len(layer))
            #Add layer indices to nodes to make adding weights to tensors faster
            index = 0
            for node in layer:
                node.l_index = index
                index += 1
        if layer_counts[0] != self.experiment.inputs:
            assert False
        return nodes, weights, layer_counts
        
    def rebuildModel(self):
        self.retraceLayers()
        working_nodes, working_weights, layer_counts = self.buildShadowElements()
        if layer_counts[-1] > self.experiment.outputs:
            for n in working_nodes[-1]:
                print("help!", n.type)
                for l in working_weights:
                    for w in l:
                        if w.origin == n:
                            print("connection starting at last layer")
                        if w.to == n:
                            print("connection ending at last layer")
        tensor_list = []
        prev_layer = 0
        curr_layer = 1
        while curr_layer < self.layers:
            weight_tensor = torch.zeros(layer_counts[curr_layer-1], layer_counts[curr_layer])
            bias_tensor = torch.zeros(layer_counts[curr_layer])
            for weight in working_weights[curr_layer]:
                if weight.to.l_index >=  layer_counts[curr_layer]:
                    print("Destination OOR: ", layer_counts, weight.to.l_index, curr_layer)
                    for l in working_nodes:
                        print("layer")
                        for n in l:
                            print(n.l_index, n.layer)
                if weight.origin.l_index >=  layer_counts[curr_layer-1]:
                    print("Origin OOR: ", weight.origin.l_index, weight.origin.layer, curr_layer-1)
                    for l in working_nodes:
                        print("layer")
                        for n in l:
                            print(n.l_index, n.layer)
                weight_tensor[weight.origin.l_index][weight.to.l_index] = weight.value
            for node in working_nodes[curr_layer]:
                bias_tensor[node.l_index] = node.bias
            tensor_list.append(weight_tensor)
            tensor_list.append(bias_tensor)
            prev_layer += 1
            curr_layer += 1
            if curr_layer >= len(self.nodes):
                break
        model = Genome_network(tensor_list, self.device)
        self.model = model.to(torch.device(self.device))
        for w in self.weights:
            w.origin = w.origin_copy
        
    #Also placeholder until I write this
    def crossover(self, other):
        return self.mutate()
        
    #Placeholder mutate so I can test other things first
    def mutate(self, innovation_num=0):
        #add one new node x percent of the time
        #add new connection y percent of the time
        new = NEATGenome(self.experiment, False)
        new.nodes = self.nodes
        new.weights = self.weights
        for node in new.nodes:
            if node.type == 'input' and node.layer != 0:
                    for w in self.weights:
                        if w.to.type == 'input':
                            print("Input as a destination!")
                    assert False
        for weight in new.weights:
            if random.random() < new.m_weight_chance:
                weight.value += (random.random() * self.mutate_effect) - (self.mutate_effect/2)
        for node in new.nodes:
            if random.random() < new.m_weight_chance and node.type != 'input': #input bias is never used, so this could be removed I guess?
                node.bias += (random.random() * self.mutate_effect) - (self.mutate_effect/2)
        if random.random() < new.m_connection_chance:
            #Select two nodes
            node_1 = new.nodes[random.randrange(len(new.nodes))]
            node_2 = new.nodes[random.randrange(len(new.nodes))]
            #Confirm they are unique and are not both in a fixed(I/O) layer
            while (node_1 == node_2) or ((node_1.layer == node_2.layer) and (node_1.type == 'input' or node_1.type == 'output')):
                node_2 = new.nodes[random.randrange(len(new.nodes))]
            if node_1.layer <= node_2.layer:
                new_weight = NEATWeight(node_1, node_2, random.random()*self.experiment.inputs)
                new.weights.append(new_weight)
                if node_2.type == 'input':
                    print("attempted to make a connection to an input", node_1.layer, node_1.type, node_2.layer, node_2.type)
                    assert False
            else:
                new_weight = NEATWeight(node_2, node_1, random.random()*self.experiment.inputs)
                new.weights.append(new_weight)
                if node_1.type == 'input':
                    print("attempted to make a connection to an input")
                    assert False
        if random.random() < new.m_node_chance:
            to_change = new.weights[random.randrange(len(new.weights))]
            new_node = NEATNode(innovation_num, 'hidden',  to_change.origin.layer + 1, random.random()*self.experiment.inputs)
            new_weight = NEATWeight(to_change.origin, new_node, random.random()*self.experiment.inputs)
            to_change.origin = new_node
            to_change.origin_copy = new_node
            new.nodes.append(new_node)
            new.weights.append(new_weight)
        new.env = self.env
        for node in new.nodes:
            if node.type == 'input' and node.layer != 0:
                    for w in self.weights:
                        if w.to.type == 'input':
                            print("Input as a destination!")
                    assert False
        new.rebuildModel()
        return new
            
        
        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
            
        
    