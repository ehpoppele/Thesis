import gym
import torch
import torch.nn as nn
import random
import copy

from genome import *

#Class representing the nodes in a NEAT network
#Go into the first part of the genotype

class NEATNode():

    def __init__(self, type, layer, bias=0, innovation_num=-1): #Innovation number default is only used for "shadow nodes"
        self.type = type #Is a string; 'input', 'output', or 'hidden' or 'shadow'
        self.layer = layer
        self.bias = bias
        self.l_index = 0 #Layer index, used only for rebuilding model faster
        self.innovation_num = innovation_num
        
class NEATWeight():

    def __init__(self, origin, to, value, innovation_num=-1):
        self.origin = origin #I hope this is a shallow copy
        self.to = to #I basically want these as pointers
        self.value = value
        #Change origin copy to origin backup or something
        self.origin_copy = origin #Hold onto for when the origin is changed to a shadow node, should still be a shallow copy though
        self.innovation_num = innovation_num
        

class NEATGenome(Genome):

    #Similar to regular init, but has 2 parts to genotype
    def __init__(self, experiment, randomize=True):
        if experiment.NODE_INNOVATION_NUMBER == -1:
            experiment.NODE_INNOVATION_NUMBER = experiment.inputs + experiment.outputs + 1
        if experiment.WEIGHT_INNOVATION_NUMBER == -1:
            experiment.WEIGHT_INNOVATION_NUMBER = experiment.inputs*experiment.outputs + 1
        self.experiment = experiment
        self.fitness = float("-inf") 
        self.mutate_effect = experiment.mutate_effect
        self.m_weight_chance = experiment.mutate_odds[0]
        self.perturb_weight_chance = experiment.mutate_odds[1]
        self.reset_weight_chance = 1.0 - experiment.mutate_odds[1]
        self.m_node_chance = experiment.mutate_odds[2]
        self.m_connection_chance = experiment.mutate_odds[3]
        self.model = None
        self.device = experiment.device
        self.env = None 
        self.layers = 2 #number of layers; I should rename this
        self.nodes = []
        self.weights = []
        self.disabled = [] #holds all the connections (weights) that have been disabled, which may be re-enabled later
        self.species = 0
        #Randomize should only be used for genomes created at the start of the experiment
        #For this reason, they have standardized innovation numbers
        if randomize:
            self.env = experiment.env
            inputs = []
            outputs = []
            for i in range(experiment.inputs):
                new_node = NEATNode('input', 0, 0, i+1)
                inputs.append(new_node)
            for i in range(experiment.outputs):
                new_node = NEATNode('output', 1, random.random()*experiment.inputs, i+1+experiment.inputs)
                outputs.append(new_node)
            j = 0
            for i in inputs: #Should clarify these vars
                k = 0
                for o in outputs:
                    new_weight = NEATWeight(i, o, random.random()*experiment.inputs, (j*experiment.outputs) + k + 1)
                    self.weights.append(new_weight)
                    k += 1
                j += 1
            self.nodes += inputs
            self.nodes += outputs
            self.rebuildModel()
            
    #Prints out all the nodes and connections for debugging
    def printToTerminal(self):
        print("Nodes:")
        for node in self.nodes:
            print(node.innovation_num, node.type, node.layer, node.bias)
        print("Weights:")
        for weight in self.weights:
            print(weight.innovation_num, weight.origin.innovation_num, weight.to.innovation_num, weight.value)
            
    #Returns if the genome is the same species as the other
    #Follows the speciation formula from the paper
    def speciesDistance(self, other):
        #Here making the assumption that nodes/weights are sorted by innovation number; I think this is true but not sure
        primary = self
        secondary = other
        #Primary is the genome with the greatest innovation number on one of its weights
        if self.weights[len(self.weights)-1].innovation_num < other.weights[len(other.weights)-1].innovation_num:
            primary = other
            secondary = self
        c1 = self.experiment.species_c1
        c2 = self.experiment.species_c2
        c3 = self.experiment.species_c3
        E = 0
        D = 0
        W = 0
        for weight in primary.weights:
            #Count all weights that are excess of secondary's greatest i nums
            if weight.innovation_num > secondary.weights[len(secondary.weights)-1].innovation_num:
                E += 1
            disjoint = True
            #Check for disjoint genes and add to weight difference
            for weight_2 in secondary.weights:
                if weight.innovation_num == weight_2.innovation_num:
                    W += abs(weight.value - weight_2.value)
                    disjoint == False
                    break
            if disjoint:
                D += 1
        for d1 in primary.disabled:
            for d2 in secondary.disabled:
                if d1.innovation_num == d2.innovation_num:
                    W += abs(d1.value - d2.value)
                    break
        N = max(len(self.nodes), len(other.nodes))
        gamma = (c1*E*c2*D)/N +(c3*W)
        return gamma
    
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
                shadowNode = NEATNode('shadow', weight.origin.layer + 1 + i)
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
        child = NEATGenome(self.experiment, False)
        child.env = self.env
        #Get higher fitness genome to be the primary parent
        primary = self
        secondary = other
        if other.fitness > self.fitness:
            primary = other
            secondary = self
        #Iterate over primary parent; we copy any nodes which it has that the other parent doesn't
        #For ones they both have, we select at random
        for node in primary.nodes:
            i_num = node.innovation_num
            other_node = next((n for n in secondary.nodes if n.innovation_num == i_num), None)
            if other_node is not None and random.random() > 0.5:
                child.nodes.append(copy.deepcopy(other_node))
            else:
                child.nodes.append(copy.deepcopy(node))
        #Do the same for weights
        for weight in primary.weights:
            i_num = weight.innovation_num
            other_weight = next((n for n in secondary.weights if n.innovation_num == i_num), None)
            if other_weight is not None and random.random() > 0.5:
                child.weights.append(copy.deepcopy(other_weight))
            else:
                child.weights.append(copy.deepcopy(weight))
        child.rebuildModel()
        return child
        
    #May need to change to make more reliable; currently allows connections to be thrown in just about anywhere
    def mutate(self, innovation_num=0):
        #add one new node x percent of the time
        #add new connection y percent of the time
        new = NEATGenome(self.experiment, False)
        new.nodes = []
        #Deep copy the parent nodes
        for node in self.nodes:
            new.nodes.append(copy.deepcopy(node))
        new.weights = []
        for weight in self.weights:
            #For weights we also need to adjust the pointers (from parent nodes to copied nodes)
            copied = copy.deepcopy(weight)
            for node in new.nodes:
                if node.innovation_num == copied.origin.innovation_num:
                    copied.origin = node
                    copied.origin_copy = node
                if node.innovation_num == copied.to.innovation_num:
                    copied.to = node
            new.weights.append(copied)
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
        if random.random() < new.m_connection_chance: #how does this work when trying to add something that already exists?
            #Select two nodes
            node_1 = new.nodes[random.randrange(len(new.nodes))]
            node_2 = new.nodes[random.randrange(len(new.nodes))]
            #Confirm they are unique and are not both in a fixed(I/O) layer
            while (node_1 == node_2) or ((node_1.layer == node_2.layer) and (node_1.type == 'input' or node_1.type == 'output')):
                node_2 = new.nodes[random.randrange(len(new.nodes))]
            if node_1.layer <= node_2.layer:
                new_weight = NEATWeight(node_1, node_2, random.random()*self.experiment.inputs, self.experiment.WEIGHT_INNOVATION_NUMBER)
                self.experiment.WEIGHT_INNOVATION_NUMBER += 1
                new.weights.append(new_weight)
                if node_2.type == 'input':
                    print("attempted to make a connection to an input", node_1.layer, node_1.type, node_2.layer, node_2.type)
                    assert False
            else:
                new_weight = NEATWeight(node_2, node_1, random.random()*self.experiment.inputs, self.experiment.WEIGHT_INNOVATION_NUMBER)
                self.experiment.WEIGHT_INNOVATION_NUMBER += 1
                new.weights.append(new_weight)
                if node_1.type == 'input':
                    print("attempted to make a connection to an input")
                    assert False
            #print("mutating new connection")
        if random.random() < new.m_node_chance:
            to_change = new.weights[random.randrange(len(new.weights))]
            disabled_connection = copy.deepcopy(to_change)
            new_node = NEATNode('hidden',  to_change.origin.layer + 1, random.random()*self.experiment.inputs, self.experiment.NODE_INNOVATION_NUMBER)
            self.experiment.NODE_INNOVATION_NUMBER += 1
            new_weight = NEATWeight(to_change.origin, new_node, 1, self.experiment.WEIGHT_INNOVATION_NUMBER)
            self.experiment.WEIGHT_INNOVATION_NUMBER += 1
            to_change.origin = new_node
            to_change.origin_copy = new_node
            new.nodes.append(new_node)
            new.weights.append(new_weight)
            new.disabled.append(disabled_connection)
            #print("mutating new node")
        new.env = self.env
        for node in new.nodes:
            if node.type == 'input' and node.layer != 0:
                    for w in self.weights:
                        if w.to.type == 'input':
                            print("Input as a destination!")
                    assert False
        new.rebuildModel()
        return new
            
        
        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
            
        
    