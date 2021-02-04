#Very basic early test file for genome class

from genome import *
import gym

experiment = {"name" : 'CartPole-v0', "inputs" : 4, "outputs" : 2, "layers" : 1, "layer_size" : 8}
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    new_net = Genome(experiment)
    fit = new_net.evalFitness()
    print(fit)
    new_net.mutate()
    fit = new_net.evalFitness()
    print(fit)