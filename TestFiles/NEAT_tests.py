from genome_NEAT import *
import experiments
import copy
import random

torch.set_default_tensor_type(torch.DoubleTensor)

experiment = experiments.cart_NEAT
experiment.NODE_INNOVATION_NUMBER = -1
experiment.WEIGHT_INNOVATION_NUMBER = -1
net = NEATGenome(experiment)
init_fit = 0
for _ in range(500):
    init_fit += net.evalFitness()
init_fit = init_fit/500
print("Initial fitness: ", init_fit)
#Mutate new net with additional node; should be functionally identical, though I will double check paper if this is intended (I think it is)
new = NEATGenome(net.experiment, False)
new.nodes = []
#Deep copy the parent nodes
for node in net.nodes:
    new.nodes.append(copy.deepcopy(node))
new.weights = []
for weight in net.weights:
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
            for w in net.weights:
                if w.to.type == 'input':
                    print("Input as a destination!")
            assert False
to_change = new.weights[random.randrange(len(new.weights))]
new_node = NEATNode('hidden',  to_change.origin.layer + 1, random.random()*net.experiment.inputs, net.experiment.NODE_INNOVATION_NUMBER)
net.experiment.NODE_INNOVATION_NUMBER += 1
new_weight = NEATWeight(to_change.origin, new_node, 1, net.experiment.WEIGHT_INNOVATION_NUMBER)
net.experiment.WEIGHT_INNOVATION_NUMBER += 1
to_change.origin = new_node
to_change.origin_copy = new_node
new.nodes.append(new_node)
new.weights.append(new_weight)
init_fit = 0
new.env = net.env
for node in new.nodes:
    if node.type == 'input' and node.layer != 0:
            for w in net.weights:
                if w.to.type == 'input':
                    print("Input as a destination!")
            assert False
new.rebuildModel()

for _ in range(500):
    init_fit += new.evalFitness()
init_fit = init_fit/500
print("Fitness with new node: ", init_fit)
            
""" Test this after; should be definite but small changes after adding one new connection
#Probably should wait to test this until after fixed; could also test as it is currently to see how it changes when it adds a connection that already exists
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
            print("mutating new connection")
"""