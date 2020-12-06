import gym
import torch
import torch.nn as nn
import random
import time
import multiprocessing

def nextStep(action, env, queue):
    observation, reward, done, _ = env.step(action)
    queue.put([(observation, reward, done), env])
    print("done")
    
"""
def retNextStep():
    def nextStep(action, env, queue):
        print("starting")
        observation, reward, done, _ = env.step(action)
        queue.put([(observation, reward, done), env])
        print("done")
    return nextStep
"""

class Genome():

    def __init__(self, experiment, randomize=True):
        self.experiment = experiment
        self.genotype = [] #Genotype will be filled with at least 2 entries, and will always be pairs of weight/bias tensors
        self.fitness = float("-inf") #default fitness is minimum before any evaluation
        self.mutate_effect = experiment.mutate_effect
        self.model = None
        self.device = experiment.device
        self.env = None #gym.make(self.experiment.name, frameskip=4)
        #Now initialize the random genotype
        if randomize:
            self.env = experiment.env
            if experiment.layers == 0:
                self.genotype.append((torch.randn(experiment.inputs, experiment.outputs)) * (1/experiment.inputs)) #Weights
                self.genotype.append(torch.zeros(experiment.outputs)) #Bias
            else:
                self.genotype.append(torch.randn(experiment.inputs, experiment.layer_size) * (1/experiment.inputs))
                self.genotype.append(torch.zeros(experiment.layer_size))
                for i in range (experiment.layers - 1):
                    self.genotype.append(torch.randn(experiment.layer_size, experiment.layer_size) * (1/experiment.layer_size))
                    self.genotype.append(torch.zeros(experiment.layer_size))
                self.genotype.append((torch.randn(experiment.layer_size, experiment.outputs)) * (1/experiment.layer_size))
                self.genotype.append(torch.zeros(experiment.outputs))
            self.rebuildModel()

    def rebuildModel(self):
        model = Genome_network(self.genotype, self.device)
        self.model = model.to(torch.device(self.device))
    
    def evalFitness(self, render=False):
        sum_reward = 0
        trials = self.experiment.trials
        for _ in range(trials):
            env = self.env
            observation = env.reset()
            for t in range(20000):
                if render:
                    time.sleep(0.02)
                    env.render()
                inputs = torch.from_numpy(observation)
                inputs = (inputs.double()).to(torch.device(self.device))
                outputs = self.model(inputs)
                action = (torch.max(outputs, 0)[1].item())
                observation, reward, done, _ = env.step(action)
                sum_reward += reward
                if done:
                    break
            env.close()
        self.fitness = sum_reward/trials
        return sum_reward/trials
    
    """
    def evalFitness(self, render=False):
        sum_reward = 0
        trials = self.experiment.trials
        for _ in range(trials):
            env = self.env
            observation = env.reset()
            for t in range(20000):
                if render:
                    time.sleep(0.02)
                    env.render()
                inputs = torch.from_numpy(observation)
                inputs = (inputs.double()).to(torch.device(self.device))
                outputs = self.model(inputs)
                action = (torch.max(outputs, 0)[1].item())
                print("starting")
                #step_func = retNextStep()
                thread_results = multiprocessing.Queue()
                process = multiprocessing.Process(target=nextStep, args=(action, env, thread_results))
                process.start()
                process.join()
                results = thread_results.get()
                print(results)
                vals = results[0]
                observation = vals[0]
                reward = vals[1]
                done = vals[2]
                env = results[1]
                sum_reward += reward
                if done:
                    break
            env.close()
        self.fitness = sum_reward/trials
        return sum_reward/trials
    """
    
    """
    #Both sets and returns the new fitness of the model
    def evalFitness(self):
        sum_reward = 0
        trials = self.experiment.trials
        for _ in range(trials):
            env = self.env #gym.make(self.experiment.name, frameskip=4)
            observation = env.reset()
            for t in range(20000): #Should this just be while true?
                inputs = torch.from_numpy(observation)
                inputs = (inputs.double()).to(torch.device(self.device))
                outputs = self.model(inputs)
                action = 0
                highest = 0
                ""
                #It seems this is an atrociously slow change??
                for i in range(len(outputs)):
                    if outputs[i] > highest:
                        highest = outputs[i]
                        action = i
                    #if highest > 0.5:
                    #    break
                ""
                action = 0
                rand_select = random.random()
                #Select an index i based on the output array distribution
                for i in range(len(outputs)):
                    rand_select -= outputs[i]
                    if rand_select < 0:
                        action = i
                        break
                observation, reward, done, _ = env.step(action)
                sum_reward += reward
                if done:
                    break
            #env.close()
        self.fitness = sum_reward/trials
        return sum_reward
    """

    #mutates based on mutation rate given by experiment
    def mutate(self):
        new = Genome(self.experiment, False) #Not creating initial random, so new.genotype is an empty list
        new.env = self.env
        for i in range(len(self.genotype)):
            new.genotype.append(self.genotype[i] + (torch.randn(self.genotype[i].size()) * (self.mutate_effect)))
        new.rebuildModel()
        return new

class Genome_network(nn.Module):

    def __init__(self, genotype, device):
        super().__init__()
        self.device = device
        cuda_genes = []
        for g in genotype:
            cuda_genes.append(g.to(torch.device(self.device)))
        self.genotype = cuda_genes

    def forward(self, inputs):
        for i in range(len(self.genotype)//2): #int division, but genotype should always be even length
            relu = nn.ReLU()
            inputs = relu((inputs @ self.genotype[2*i]) + self.genotype[2*i + 1])
            #inputs.to(torch.device(self.device)) #Is this needed?
        soft = nn.Softmax(dim=0)
        return soft(inputs)
