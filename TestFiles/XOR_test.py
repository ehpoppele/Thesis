#An attempt to build a correct/optimal network for the XOR circuit problem
#Due to the way in which my networks are constructed and run as matrices rather than plain connections, the ideal solution the NEAT paper uses is impossible
#While I was unable to find an optimal (fitness near 4.0) solution, I could hand-craft a network with fitness > 3.7
#NEAT was mostly stuck with 3.5 as the maximum
#Since this problem is inherently different from the one in the paper, despite my best efforts, I have abandoned it as a benchmark for NEAT

import experiments
import torch
from genome_NEAT import *

experiment = experiments.xor

torch.set_default_tensor_type(torch.DoubleTensor)

experiment.NODE_INNOVATION_NUMBER = -1
experiment.WEIGHT_INNOVATION_NUMBER = -1

optimal_net = NEATGenome(experiment, False)
node1 = NEATNode('input', 0, 0, 1)
node2 = NEATNode('input', 0, 0, 2)
node3 = NEATNode('output', 0, -1, 3)
node4 = NEATNode('hidden', 0, -1.5, 4)
node5 = NEATNode('hidden', 0, -0.5, 5)
node6 = NEATNode('hidden', 0, -0.5, 6)
weight1 = NEATWeight (node1, node5, 1, 1)
weight6 = NEATWeight (node5, node3, 5, 6)
weight2 = NEATWeight (node2, node6, 1, 2)
weight7 = NEATWeight (node6, node3, 5, 7)
weight3 = NEATWeight (node1, node4, 1, 3)
weight4 = NEATWeight (node2, node4, 1, 4)
weight5 = NEATWeight (node4, node3, -10, 5)
optimal_net.nodes = [node1, node2, node3, node4, node5, node6]
optimal_net.weights = [weight1, weight2, weight3, weight4, weight5, weight6, weight7]
optimal_net.env = optimal_net.experiment.env
optimal_net.rebuildModel()
for g in optimal_net.model.genotype:
    print(g)

optimal_net.evalFitness(True)

print(optimal_net.fitness)
optimal_net.printToTerminal()