import numpy

class XOR_env():
    
    def __init__(self):
        self.values = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]]
        self.iteration = 0
        self.is_rendered = False
        
    def reset(self):
        self.iteration = 0
        if self.is_rendered:
            print("Now checking on input: (0,0)")
        return numpy.array(self.values[self.iteration][0])
        
    #Maybe change this later, but just need a function so there aren't errors if render is called
    def render(self):
        #print("XOR cannot render")
        self.is_rendered = True
        
    def close(self):
        self.iteration = 0
    
    #Checks if action matches the last output for correct XOR behavior, then returns the next pair of bits
    def step(self, action):
        done = False
        obs = numpy.array([0,0])
        difference = abs(action - self.values[self.iteration][1])
        reward = 1/(difference+1)
        self.iteration += 1
        if self.iteration > 3:
            done = True
        else:
            obs = numpy.array(self.values[self.iteration][0])
        if self.is_rendered:
            print("Network output was: " + str(action))
            print("Now checking on input: (" + str(obs[0]) + "," + str(obs[1]) + ")")
        return (obs, reward, done, 0) #not sure if tuple or list? also forgot what the last thing is but need a placeholder