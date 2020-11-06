#Defines the experiment class, which holds params for experiments that are accessed by the GA
#Also creates experiment instances and assigns the appropriate values to them
import gym #currently here so I can create envs associated with each experiment
#This lets me do things like frameskip here, and constructor params that vary by more than name without case checking in genome
#Think this should be fine? genomes then just copy env I believe

class Experiment():

    def __init__(self, name):
        self.name = name
        self.outfile = 'terminal' #default output writes to terminal rather than a file
        self.genome = 'Basic'

#Cart pole config, pretty much solves the problem
#might not be optimal, but final solution about 200 (max score) each time
cart_pole = Experiment('CartPole-v0')
cart_pole.env = gym.make('CartPole-v0')
cart_pole.device        = 'cuda'
cart_pole.inputs        = 4
cart_pole.outputs       = 2
cart_pole.layers        = 4 #seems that only 2 are necessary?
cart_pole.layer_size    = 4
cart_pole.trials        = 10
cart_pole.population    = 51
cart_pole.generations   = 35    #needs 35 for stable no-oscillation
cart_pole.child_count   = 0     #Experiment is basic GA, so no crossover
cart_pole.mutate_range  = 10
cart_pole.mutate_count  = cart_pole.population - 1
cart_pole.mutate_effect = 10/4
cart_pole.elite_count   = cart_pole.population - (cart_pole.child_count + cart_pole.mutate_count)
cart_pole.elite_range   = 10
cart_pole.elite_evals   = 30
#---------------------------
#Frostbite
frostbite_1 = Experiment('Frostbite-ram-v0')
frostbite_1.env             = gym.make('Frostbite-ram-v0', frameskip=4)
frostbite_1.device          = 'cpu'
frostbite_1.inputs          = 128
frostbite_1.outputs         = 18
frostbite_1.layers          = 2
frostbite_1.layer_size      = 256
frostbite_1.trials          = 1
frostbite_1.population      = 1001
frostbite_1.generations     = 10
frostbite_1.child_count     = 0
frostbite_1.mutate_range    = 20
frostbite_1.mutate_count    = frostbite_1.population - 1
frostbite_1.mutate_effect   = 0.002
frostbite_1.elite_count     = frostbite_1.population - (frostbite_1.child_count + frostbite_1.mutate_count)
frostbite_1.elite_range     = 10
frostbite_1.elite_evals     = 30
frostbite_1.outfile         = 'terminal'
#---------------------------
#Venture
venture_1 = Experiment('Venture-ram-v0')
venture_1.env             = gym.make('Venture-ram-v0', frameskip=4)
venture_1.device          = 'cuda'
venture_1.inputs          = 128
venture_1.outputs         = 18
venture_1.layers          = 2
venture_1.layer_size      = 512
venture_1.trials          = 1
venture_1.population      = 1001
venture_1.generations     = 10
venture_1.child_count     = 0
venture_1.mutate_range    = 20
venture_1.mutate_count    = venture_1.population - 1
venture_1.mutate_effect   = 0.002
venture_1.elite_count     = venture_1.population - (venture_1.child_count + venture_1.mutate_count)
venture_1.elite_range     = 10
venture_1.elite_evals     = 30
#---------------------------
c_p_NEAT = cart_pole
c_p_NEAT.env         = gym.make('CartPole-v0')
c_p_NEAT.genome      = 'NEAT'
c_p_NEAT.mutate_odds = [0.7, 0.3, 0.4] #Percent of time that a mutation will occur in the mutate function
#First is odds for each weight to change, second is odds to add 1 node, third is to add 1 connection

