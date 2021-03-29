import gym
import torch
import torch.nn as nn
import random

from genome import *

#Class representing the nodes in a NEAT network
#Tracks layer and bias
class NEATNode():

    def __init__(self, type, layer, bias=0.0, innovation_num=-1): #Innovation number default is only used for "shadow nodes"
        self.type = type #Is a string; 'input', 'output', or 'hidden' or 'shadow'
        self.layer = layer
        self.bias = bias
        self.l_index = 0 #Layer index, used only for rebuilding model faster
        self.innovation_num = innovation_num
        
    #Returns a new node with the same values as this one
    def newCopy(self):
        new = NEATNode(self.type, self.layer, self.bias, self.innovation_num)
        new.l_index = self.l_index
        return new
        
#Class to track the weights/connections in the network
#Tracks the value of the weight and what it connects
class NEATWeight():

    def __init__(self, origin, destination, value=1.0, innovation_num=-1):
        self.origin = origin #Shallow copy/pointer
        self.destination = destination
        self.value = value
        self.origin_backup = origin #Hold onto for when the origin is changed to a shadow node when tracing the network; should still be a shallow copy though
        self.innovation_num = innovation_num
        
    #Returns a new weight with the same values, including origin/destination
    def newCopy(self):
        new = NEATWeight(self.origin, self.destination, self.value, self.innovation_num)
        new.origin_backup = self.origin_backup
        return new
        
#Actual genome class; overrides most methods from the other class
class NEATGenome(Genome):

    #Similar to regular init, but has 2 parts to genotype for nodes and weights
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
        self.reenable_chance = experiment.reenable_chance
        self.model = None
        self.device = experiment.device
        self.env = None 
        self.layers = 2 #number of layers; I should rename this
        self.nodes = []
        self.weights = []
        self.disabled = [] #holds all the connections (weights) that have been disabled, which may be re-enabled later
        #Randomize should only be used for genomes created at the start of the experiment
        #For this reason, they have standardized (sequential) innovation numbers
        if randomize:
            self.env = experiment.env
            inputs = []
            outputs = []
            for i in range(experiment.inputs):
                new_node = NEATNode('input', 0, 0, i+1)
                inputs.append(new_node)
            for i in range(experiment.outputs):
                new_node = NEATNode('output', 1, random.random()*self.mutate_effect, i+1+experiment.inputs)
                outputs.append(new_node)
            j = 0
            for i in inputs: #Should clarify these vars
                k = 0
                for o in outputs:
                    new_weight = NEATWeight(i, o, random.random()*self.mutate_effect, (j*experiment.outputs) + k + 1)
                    self.weights.append(new_weight)
                    k += 1
                j += 1
            self.nodes += inputs
            self.nodes += outputs
            #self.rebuildModel()
            
    def newCopy(self):
        new = NEATGenome(self.experiment, False)
        new.fitness = self.fitness
        new.env = self.env
        new.layers = self.layers
        for n in self.nodes:
            copy = n.newCopy()
            new.nodes.append(copy)
        for w in self.weights:
            copy = w.newCopy()
            new.weights.append(copy)
        for d in self.disabled:
            copy = d.newCopy()
            new.disabled.append(copy)
        #Same loop as in crossover to confirm everything is pointing to the right place
        for weight in new.weights:
            assigned = 0
            for node in new.nodes:
                if weight.origin.innovation_num == node.innovation_num:
                    weight.origin = node
                    assigned += 1
                if weight.destination.innovation_num == node.innovation_num:
                    weight.destination = node
                    assigned += 1
            if assigned != 2:
                print("Issue with copying; missing a node in the copy")
                assert False
        for weight in new.disabled:
            assigned = 0
            for node in new.nodes:
                if weight.origin.innovation_num == node.innovation_num:
                    weight.origin = node
                    assigned += 1
                if weight.destination.innovation_num == node.innovation_num:
                    weight.destination = node
                    assigned += 1
            if assigned != 2:
                print("Issue with copying; missing a node in the copy")
                assert False
        return new
            
    #Prints out all the nodes and connections for debugging
    def printToTerminal(self):
        print("Fitness:")
        print(self.fitness)
        print("Extra Nodes:")
        print(len(self.nodes)-self.experiment.inputs-self.experiment.outputs) 
        #for node in self.nodes:
        #    print(node.innovation_num, node.type, node.layer, node.bias)
        print("Extra Weights:")
        print(len(self.weights) - (self.experiment.inputs*self.experiment.outputs))
        #for weight in self.weights:
        #    print(weight.innovation_num, weight.origin.innovation_num, weight.destination.innovation_num, weight.value)
        print("Disabled Weights:")
        print(len(self.disabled))
        #for weight in self.disabled:
        #    print(weight.innovation_num, weight.origin.innovation_num, weight.destination.innovation_num, weight.value)
        print()
            
    #Returns the species distance between two genomes, following the formula from the paper
    def speciesDistance(self, other):
        #Here making the assumption that nodes/weights are sorted by innovation number; I think this is true but not sure
        primary = self
        secondary = other
        #Primary is the genome with the greatest innovation number on one of its weights
        if self.weights[-1].innovation_num < other.weights[-1].innovation_num:
            primary = other
            secondary = self
        c1 = self.experiment.species_c1
        c2 = self.experiment.species_c2
        c3 = self.experiment.species_c3
        E = 0 #excess
        D = 0 #disjoint
        W = 0 #average weight differences
        #Now we do a clever algorithm to find disjoint/excess weights; should be linear since the lists are sorted
        primary_index = 0
        secondary_index = 0
        while (primary_index < len(primary.weights) and secondary_index < len(secondary.weights)):
            if primary.weights[primary_index].innovation_num < secondary.weights[secondary_index].innovation_num:
                D +=1
                primary_index += 1
            elif primary.weights[primary_index].innovation_num == secondary.weights[secondary_index].innovation_num:
                W += abs(primary.weights[primary_index].value - secondary.weights[secondary_index].value)
                primary_index += 1
                secondary_index += 1
            elif primary.weights[primary_index].innovation_num > secondary.weights[secondary_index].innovation_num:
                secondary_index += 1
        while (primary_index < len(primary.weights)):
            E += 1
            primary_index += 1
        #Next we loop through the nodes, first reassigning based on which has more
        if primary.nodes[-1].innovation_num < secondary.nodes[-1].innovation_num:
            primary = secondary
            secondary = primary
        primary_index = 0
        secondary_index = 0
        while (primary_index < len(primary.nodes) and secondary_index < len(secondary.nodes)):
            if primary.nodes[primary_index].innovation_num < secondary.nodes[secondary_index].innovation_num:
                D +=1
                primary_index += 1
            elif primary.nodes[primary_index].innovation_num == secondary.nodes[secondary_index].innovation_num:
                W += abs(primary.nodes[primary_index].bias - secondary.nodes[secondary_index].bias)
                primary_index += 1
                secondary_index += 1
            elif primary.nodes[primary_index].innovation_num > secondary.nodes[secondary_index].innovation_num:
                secondary_index += 1
        while (primary_index < len(primary.nodes)):
            E += 1
            primary_index += 1
        #And another loop for the disabled weights
        primary_index = 0
        secondary_index = 0
        while (primary_index < len(primary.disabled) and secondary_index < len(secondary.disabled)):
            if primary.disabled[primary_index].innovation_num < secondary.disabled[secondary_index].innovation_num:
                primary_index += 1
            elif primary.disabled[primary_index].innovation_num == secondary.disabled[secondary_index].innovation_num:
                W += abs(primary.disabled[primary_index].value - secondary.disabled[secondary_index].value)
                primary_index += 1
                secondary_index += 1
            elif primary.disabled[primary_index].innovation_num > secondary.disabled[secondary_index].innovation_num:
                secondary_index += 1
        #We don't count excess on disabled weights, so no end loop
        
        """       
        for weight in primary.weights:
            #Count all weights that are excess of secondary's greatest innovation nums
            if weight.innovation_num > secondary.weights[len(secondary.weights)-1].innovation_num:
                E += 1
            else:
                disjoint = True
                #Check for disjoint genes and add to weight difference
                for weight_2 in secondary.weights:
                    if weight.innovation_num == weight_2.innovation_num:
                        W += abs(weight.value - weight_2.value)
                        disjoint == False
                        break
                if disjoint:
                    D += 1
        #repeat for nodes
        for node in primary.nodes:
            if node.innovation_num > secondary.nodes[-1].innovation_num:
                E += 1
            else:
                disjoint = True
                for node_2 in secondary.nodes:
                    if node.innovation_num == node_2.innovation_num:
                        W += abs(node.bias - node_2.bias)
                        disjoint == False
                        break
                if disjoint:
                    D += 1
        for d1 in primary.disabled:
            for d2 in secondary.disabled:
                if d1.innovation_num == d2.innovation_num:
                    W += abs(d1.value - d2.value)
                    break
        """
        #Apply equation from NEAT paper
        N = max(len(self.nodes)+len(self.weights), len(other.nodes)+len(other.weights))
        gamma = (c1*E)/N + (c2*D)/N +(c3*W)
        return gamma
    
    #Goes through the genotype and updates the layers/depth value of each node
    #Assumes input nodes are fixed as the only nodes at layer zero
    def retraceLayers(self):
        for node in self.nodes:
            node.layer = 0
        layer = 0
        #Starting at layer zero, we loop over all the weights and find the connections out of a node at this layer
        #We then ensure the layer of the connections destination is higher than that of its origin
        #The loop ends when we can't find any connections leaving the current layer
        #This could be optimized more, I think; technically still linear since it's length of weights times number of layers
        while True:
            weights_leaving_this_layer = False
            for weight in self.weights:
                if weight.origin.layer == layer:
                    if weight.destination.layer <= layer:
                        weight.destination.layer = layer + 1
                    weights_leaving_this_layer = True
            if not weights_leaving_this_layer:
                break
            layer += 1
        #Now that all nodes should have the proper layer, we find the maximum one
        max_layer = 0
        for node in self.nodes:
            if node.layer > max_layer:
                    max_layer = node.layer
        #Set all output nodes to max layer
        for n in self.nodes:
            if n.type == 'output':
                n.layer = max_layer
        self.layers = max_layer + 1
        
    #Returns a altered genotype that includes the extra nodes necessary to make complete connections
    #For the nodes and weights to be converted effectively into tensors, we need connections to stop at each layer
    #Rather than have one running from layer 0 to 3, for example; this adds extra connections in between with a weight of 1 and bias of zero
    #up to the layer before the destination layer, then alters the original weight to go just between those last two layers
    #Also, returns nodes and weights in 2D lists, separated by layer (weights sorted by destination layer)
    #And returns a list with number of nodes in each layer
    #These in-between "shadow connections" do end up applying the activation function multiple times, so great with ReLU but not with sigmoidal
    def buildShadowElements(self):
        nodes = []
        for _ in range(self.layers):
            nodes.append([])
        for node in self.nodes:
            nodes[node.layer].append(node)
        shadowCount = 0 #Counter to track id for shadow node
        weights = []
        for _ in range(self.layers):
            weights.append([])
        #loop through all connections
        for weight in self.weights:
            prev = weight.origin
            for i in range(weight.destination.layer - (weight.origin.layer + 1)): #Add nodes as necessary
                shadowNode = NEATNode('shadow', weight.origin.layer + 1 + i)
                shadowCount += 1
                shadowWeight = NEATWeight(prev, shadowNode, 1.0)
                nodes[shadowNode.layer].append(shadowNode)
                weights[shadowNode.layer].append(shadowWeight)
                prev = shadowNode
            weight.origin = prev #!!! Need to fix this, since it needs a origin pointer and a origin id value <--- what does this mean?
            weights[weight.destination.layer].append(weight)
        layer_counts = []
        for layer in nodes:
            layer_counts.append(len(layer))
            #Add layer indices to nodes to make adding weights to tensors faster
            index = 0
            for node in layer:
                node.l_index = index
                index += 1
        if layer_counts[0] != self.experiment.inputs or layer_counts[-1] != self.experiment.outputs:
            print(layer_counts)
            self.printToTerminal()
            assert False
        return nodes, weights, layer_counts
        
    #Main function to construct tensors out of the network of connections and nodes
    def rebuildModel(self):
        self.retraceLayers() #Makes sure each node is labeled with the correct layer, as these may have shifted after crossover/mutation
        working_nodes, working_weights, layer_counts = self.buildShadowElements()
        """
        #Debugging code, will remove once im sure NEAT is stable
        if layer_counts[-1] > self.experiment.outputs:
            self.printToTerminal()
            for n in working_nodes[-1]:
                print("help!", n.type)
                for l in working_weights:
                    for w in l:
                        if w.origin == n:
                            print("connection starting at last layer")
                        if w.destination == n:
                            print("connection ending at last layer")
        """
        tensor_list = []
        prev_layer = 0
        curr_layer = 1
        #loop through and make a 2D matrix for each layer and set of bias values
        while curr_layer < self.layers:
            """
            if (layer_counts[curr_layer] <= 0):
                print("Empty layers found in the network:")
                print(layer_counts)
                self.printToTerminal()
                assert False
            """
            weight_tensor = torch.zeros(layer_counts[curr_layer-1], layer_counts[curr_layer])
            bias_tensor = torch.zeros(layer_counts[curr_layer])
            for weight in working_weights[curr_layer]:
                """
                if weight.destination.l_index >=  layer_counts[curr_layer]:
                    print("Destination OOR: ", layer_counts, weight.destination.l_index, curr_layer)
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
                """
                weight_tensor[weight.origin.l_index][weight.destination.l_index] = weight.value
            for node in working_nodes[curr_layer]:
                bias_tensor[node.l_index] = node.bias
            tensor_list.append(weight_tensor)
            tensor_list.append(bias_tensor)
            prev_layer += 1
            curr_layer += 1
            if curr_layer >= len(self.nodes):
                break
        model = GenomeNetwork(tensor_list, self.device, self.experiment)
        self.model = model.to(torch.device(self.device))
        #Now reset all the connections to have the proper origin, as it may have been shifted to a shadow node
        for w in self.weights:
            w.origin = w.origin_backup
        
    #Returns an offspring genome from combining two genomes through crossover
    #Takes all the disjoint/excess nodes and connections from the fitter of the two
    #Randomly selects between the two when they share a node or connection
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
        primary_index = 0
        secondary_index = 0
        while(primary_index < len(primary.nodes) and secondary_index < len(secondary.nodes)):
            if primary.nodes[primary_index].innovation_num < secondary.nodes[secondary_index].innovation_num:
                child.nodes.append(primary.nodes[primary_index].newCopy())
                primary_index += 1
            elif primary.nodes[primary_index].innovation_num == secondary.nodes[secondary_index].innovation_num:
                if random.random() > 0.5:
                    child.nodes.append(secondary.nodes[secondary_index].newCopy())
                else:
                    child.nodes.append(primary.nodes[primary_index].newCopy())
                primary_index += 1
                secondary_index += 1
            elif primary.nodes[primary_index].innovation_num > secondary.nodes[secondary_index].innovation_num:
                secondary_index += 1
        while (primary_index < len(primary.nodes)):
            child.nodes.append(primary.nodes[primary_index].newCopy())
            primary_index += 1
            
        #Loop for weights
        primary_index = 0
        secondary_index = 0
        while(primary_index < len(primary.weights) and secondary_index < len(secondary.weights)):
            if primary.weights[primary_index].innovation_num < secondary.weights[secondary_index].innovation_num:
                child.weights.append(primary.weights[primary_index].newCopy())
                primary_index += 1
            elif primary.weights[primary_index].innovation_num == secondary.weights[secondary_index].innovation_num:
                if random.random() > 0.5:
                    child.weights.append(secondary.weights[secondary_index].newCopy())
                else:
                    child.weights.append(primary.weights[primary_index].newCopy())
                primary_index += 1
                secondary_index += 1
            elif primary.weights[primary_index].innovation_num > secondary.weights[secondary_index].innovation_num:
                secondary_index += 1
        while (primary_index < len(primary.weights)):
            child.weights.append(primary.weights[primary_index].newCopy())
            primary_index += 1
            
        #And do the same for disabled weights, with a chance for them to re-enabled
        to_add = [] #Disabled weights must be added in a sorted order into the list to maintain the order
        primary_index = 0
        secondary_index = 0
        while(primary_index < len(primary.disabled) and secondary_index < len(secondary.disabled)):
            if primary.disabled[primary_index].innovation_num < secondary.disabled[secondary_index].innovation_num:
                child.disabled.append(primary.disabled[primary_index].newCopy())
                primary_index += 1
            elif primary.disabled[primary_index].innovation_num == secondary.disabled[secondary_index].innovation_num:
                if random.random() > 0.5:
                    if random.random() < self.reenable_chance:
                        to_add.append(secondary.disabled[secondary_index].newCopy())
                    else:
                        child.disabled.append(secondary.disabled[secondary_index].newCopy())
                else:
                    if random.random() < self.reenable_chance:
                        to_add.append(primary.disabled[primary_index].newCopy())
                    else:
                        child.disabled.append(primary.disabled[primary_index].newCopy())
                primary_index += 1
                secondary_index += 1
            elif primary.disabled[primary_index].innovation_num > secondary.disabled[secondary_index].innovation_num:
                secondary_index += 1
        while (primary_index < len(primary.disabled)):
            if random.random() < self.reenable_chance:
                to_add.append(primary.disabled[primary_index].newCopy())
            else:
                child.disabled.append(primary.disabled[primary_index].newCopy())
            primary_index += 1
        """
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
        #And do the same for disabled weights, with a chance for them to re-enabled
        to_add = [] #Disabled weights must be added in a sorted order into the list to maintain the order
        for weight in primary.disabled:
            i_num = weight.innovation_num
            other_weight = next((n for n in secondary.disabled if n.innovation_num == i_num), None)
            if other_weight is not None and random.random() > 0.5:
                if random.random() < self.reenable_chance:
                    to_add.append(copy.deepcopy(other_weight))
                else:
                    child.disabled.append(copy.deepcopy(other_weight))
            else:
                if random.random() < self.reenable_chance:
                    to_add.append(copy.deepcopy(weight))
                else:
                    child.disabled.append(copy.deepcopy(weight))
        """
        #Add re-enabled weights while keeping weights sorted by inum
        for w in to_add:
            added = False
            i_num = w.innovation_num
            for i in range(len(child.weights)):
                if child.weights[i].innovation_num == i_num:
                    added = True #This is a lie, technically
                    break #This shouldn't happen, but just in case; we don't want multiple weights with same inums
                if child.weights[i].innovation_num > i_num:
                    child.weights.insert(i, w)
                    added = True
                    break
            if not added:
                child.weights.append(w)
        #Now we need to fix the weights so they point to the new copies of the nodes now in this network
        #This looks like an expensive double loop; not sure yet how much time it actually takes up
        for weight in child.weights:
            assigned = 0
            for node in child.nodes:
                if weight.origin.innovation_num == node.innovation_num:
                    weight.origin = node
                    assigned += 1
                if weight.destination.innovation_num == node.innovation_num:
                    weight.destination = node
                    assigned += 1
            if assigned != 2:
                print("Issue with crossover; missing a node in the child")
                assert False
        for weight in child.disabled:
            assigned = 0
            for node in child.nodes:
                if weight.origin.innovation_num == node.innovation_num:
                    weight.origin = node
                    assigned += 1
                if weight.destination.innovation_num == node.innovation_num:
                    weight.destination = node
                    assigned += 1
            if assigned != 2:
                print("Issue with crossover; missing a node in the child")
                assert False
        #child.rebuildModel()
        return child
        
    #Mutation function perturbs and resets values on weights and biases and can add new nodes or connections
    #Returns a new genome and makes no changes to the original
    def mutate(self, innovation_num=0):
        new = NEATGenome(self.experiment, False)
        new.env = self.env
        new.nodes = []
        #Deep copy over the parent nodes
        for node in self.nodes:
            new.nodes.append(node.newCopy())
        new.weights = []
        for weight in self.weights:
            #For weights we also need to adjust the pointers (from parent nodes to copied nodes)
            copied = weight.newCopy()
            for node in new.nodes:
                if node.innovation_num == copied.origin.innovation_num:
                    copied.origin = node
                    copied.origin_backup = node
                if node.innovation_num == copied.destination.innovation_num:
                    copied.destination = node
            new.weights.append(copied)
        for node in new.nodes:
            if node.type == 'input' and node.layer != 0:
                    for w in self.weights:
                        if w.destination.type == 'input':
                            print("Input as a destination!")
                    assert False
        #Chance to perturb or reset each weight value independently
        for weight in new.weights:
            if random.random() < new.m_weight_chance:
                if random.random() < new.perturb_weight_chance:
                    weight.value += (random.random() * self.mutate_effect) - (self.mutate_effect/2) #perturb
                else:
                    weight.value = (random.random() * self.mutate_effect) - (self.mutate_effect/2) #reset
        #Chance to perturb or reset the bias value for each node
        for node in new.nodes:
            if random.random() < new.m_weight_chance and node.type != 'input': #input bias is never used, so this could be removed I guess?
                if random.random() < new.perturb_weight_chance:
                    node.bias += (random.random() * self.mutate_effect) - (self.mutate_effect/2) #perturb
                else:
                    node.bias = (random.random() * self.mutate_effect) - (self.mutate_effect/2) #reset
        if random.random() < new.m_connection_chance:
            #Select two nodes
            node_1 = new.nodes[random.randrange(len(new.nodes))]
            node_2 = new.nodes[random.randrange(len(new.nodes))]
            #Confirm they are unique and are not both in a fixed(I/O) layer
            while (node_1 == node_2) or ((node_1.layer == node_2.layer) and (node_1.type == 'input' or node_1.type == 'output')):
                node_2 = new.nodes[random.randrange(len(new.nodes))]
            #Quick solution to making the same connection again; just block the mutation from happening
            duplicate = False
            for w in new.weights:
                if (w.origin == node_1 and w.destination == node_2) or (w.origin == node_2 and w.destination == node_1):
                    duplicate = True
            if not duplicate:
                #Find the lower layer one or default if equal layers, then make the connection
                if node_1.layer <= node_2.layer:
                    new_weight = NEATWeight(node_1, node_2, random.random()*self.experiment.inputs, self.experiment.WEIGHT_INNOVATION_NUMBER)
                    self.experiment.WEIGHT_INNOVATION_NUMBER += 1
                    new.weights.append(new_weight)
                else:
                    new_weight = NEATWeight(node_2, node_1, random.random()*self.experiment.inputs, self.experiment.WEIGHT_INNOVATION_NUMBER)
                    self.experiment.WEIGHT_INNOVATION_NUMBER += 1
                    new.weights.append(new_weight)
        #A new node is added on an existing connection, shifting the original connection 'up', to connect the new node to the original destination node
        node_added = False
        if random.random() < new.m_node_chance:
            to_change = new.weights[random.randrange(len(new.weights))]
            disabled_connection = to_change.newCopy()
            new_node = NEATNode('hidden',  to_change.origin.layer + 1, 0, self.experiment.NODE_INNOVATION_NUMBER)
            self.experiment.NODE_INNOVATION_NUMBER += 1
            new_weight = NEATWeight(to_change.origin, new_node, 1, self.experiment.WEIGHT_INNOVATION_NUMBER)
            self.experiment.WEIGHT_INNOVATION_NUMBER += 1
            to_change.origin = new_node
            to_change.origin_backup = new_node
            to_change.innovation_num = self.experiment.WEIGHT_INNOVATION_NUMBER
            self.experiment.WEIGHT_INNOVATION_NUMBER += 1
            new.nodes.append(new_node)
            new.weights.append(new_weight)
            new.disabled.append(disabled_connection)
            node_added = True
        #new.rebuildModel()
        return new
            
