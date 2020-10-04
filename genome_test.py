from genome import *

experiment = {"name" : 'CartPole-v0', "inputs" : 4, "outputs" : 2, "layers" : 1, "layer_size" : 8}

if __name__ == "__main__":
    new_net = Genome(experiment)
    print(new_net.genotype)