#Environemt for the XOR circuit problem, built with the same callable methods that are used for gym envs in the code
#Tried to replicate as best as was described in the NEAT paper as a benchmark, but there still seem to be some issues
#Could try to adjust fitness function with steeper exponential curve towards the ideal solution
#This requires some modifications genome.py to run, as it expects the nets to run a double instead of an integer selection
#So the space of actions, in that sense, is infinite, which is a problem for the current implementation of everything else

import numpy

class XOR_env():
    
    def __init__(self):
        self.values = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]] #Correct pairs of input-output values
        self.iteration = 0 #Which pair it is currently working on
        self.is_rendered = False
        
    def reset(self):
        self.iteration = 0
        if self.is_rendered:
            print("Now checking on input: (0,0)")
        return numpy.array(self.values[self.iteration][0])
        
    #causes input/output to be printed to terminal
    def render(self):
        self.is_rendered = True
        
    def close(self):
        self.iteration = 0
    
    #Checks if action matches the last output for correct XOR behavior, then returns the next pair of bits as observation
    def step(self, action):
        done = False
        obs = numpy.array([0,0])
        difference = abs(action - self.values[self.iteration][1])
        reward = 1/(difference+1) #This could be adjusted; at present getting 3/4 gives a pretty close reward to 4/4 correct
        self.iteration += 1
        if self.iteration > 3:
            done = True
        else:
            obs = numpy.array(self.values[self.iteration][0])
        if self.is_rendered:
            print("Network output was: " + str(action))
            print("Now checking on input: (" + str(obs[0]) + "," + str(obs[1]) + ")")
        return (obs, reward, done, 0) #Last thing is thrown away by genome but still needs to be returned