import gym
import torch
import torch.nn as nn
#import the pytorch stuff, just basic and nn?

class Genome():
    
    def __init__(self, experiment):
        self.experiment = experiment
        self.genotype = [] #Genotype will be filled with at least 2 entries, and will always be pairs of weight/bias tensors
        self.fitness = float("-inf") #default fitness is minimum before any evaluation
        self.model = None
        #Now initialize the random genotype
        if experiment["layers"] == 0: #should always be at least 1, but this will catch some errors
            self.genotype.append((torch.randn(experiment["inputs"], experiment["outputs"])) * (1/experiment["inputs"])) #Weights
            self.genotype.append(torch.zeros(experiment["outputs"])) #Bias
        else:
            self.genotype.append(torch.randn(experiment["inputs"], experiment["layer_size"]) * (1/experiment["inputs"]))
            self.genotype.append(torch.zeros(experiment["layer_size"]))
            for i in range (experiment["layers"] - 1):
                self.genotype.append(torch.randn(experiment["layer_size"], experiment["layer_size"]) * (1/experiment["layer_size"]))
                self.genotype.append(torch.zeros(experiment["layer_size"]))
            self.genotype.append((torch.randn(experiment["layer_size"], experiment["outputs"])) * (1/experiment["layer_size"]))
            self.genotype.append(torch.zeros(experiment["outputs"]))
        self.rebuildModel()
            
    def rebuildModel(self):
        #There is certainly a faster way to do this, that doesn't require a python loop. I'll look into it later
        def model(inputs):
            for i in range(len(self.genotype)//2): #int division, but genotype should always be even length
                inputs = nn.relu((inputs @ self.genotype[2*i] + self.genotype[2*i + 1]))
            return nn.Softmax(inputs)
        self.model = model
        
    def evalFitness(self, experiment):
        assert (experiment == self.experiment), "Attempted to evaluate on incompatible experiment"
        env = gym.make(experiment["name"])
        observation = env.reset()
        for t in range(100): #Can work with changing this to much higher?
            env.render()
            outputs = self.model(observation)
            action = 0
            rand_select = random()
            #Select an index i based on the output array distribution
            for i in range(len(outputs)):
                rand_select -= outputs[i]
                if rand_select < 0:
                    action = i + 1
                    break
                    #We should always break at some point, since outputs should sum 1, and rand_select is 1 at max
                    #but in a rare rounding error we default to 0
            observation, reward, done, info = env.step(action)
            sum_reward += reward
            if done:
                break
        env.close()
        self.fitness = sum_reward
        return sum_reward
        
    #This currently mutates everything by a fair amount
    #Will change later to be param-based, to adjust number of tensors mutated and by how much
    def mutate(self):
        for i in range(len(self.genotype)):
            self.genotype[i] = self.genotype[i] + (torch.randn(self.genotype[i].size()) * (1/self.genotype[i].size()[0]))
        self.rebuildModel()