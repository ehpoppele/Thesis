#Defines the experiment class, which holds params for experiments that are accessed by the GA
#Also creates experiment instances and assigns the appropriate values to them
import gym #currently here so I can create envs associated with each experiment
import copy
import torch
import torch.nn as nn #For activation functions
from TestFiles.XOR_env import * #Custom environment; has similar functions to a gym env but more limited since I only need a few functions

#This lets me do things like frameskip here, and constructor params that vary by more than name without case checking in genome
#Think this should be fine? genomes then just copy env I believe

class Experiment():

    def __init__(self, name):
        self.name = name
        self.outfile = 'terminal' #default output writes to terminal rather than a file
        self.genome = 'Basic'
        self.genome_file = None
        self.fitness_sharing = False
        self.species_select = 'range'
        self.thread_count = 1
        self.device = 'cpu'
        self.population = 101
        self.generations = 100
        self.gens_to_improve = self.generations #To effectively ignore species constraints
        self.max_species_dist =  1.0 #Basic genomes should always be considered same species, but this is dealt with in distance function
        self.interspecies_crossover = 0.0 #No crossover in general
        self.mutate_ratio = 1.0 #All offspring from mutation, none from crossover
        self.mutate_range = 10
        self.mutate_effect = 1.0 #Size of the mutations that occur; uniform random with this range centered on zero
        self.elite_per_species   = 1 #Only 1 species, but only 1 genome copied by default
        self.elite_threshold   = 1 #Should only ever be 1 species
        self.elite_range   = 10 #number of genomes checked for elite copy over (should be <= threshold, always)
        self.elite_evals   = 30 #Evaluations done on each genome in the elite range to find a more accurate fitness value
        self.activation_func = nn.ReLU()
        self.activation_const = 1.
        self.thread_count = 1
        
class NEATExperiment(Experiment):
    
    def __init__(self, name):
        self.name = name
        self.outfile = 'terminal' #file to write output to; default simply prints to terminal
        self.genome = 'NEAT'
        self.genome_file = None #File to save top genomes to through pickling
        self.fitness_sharing = True #Fitness sharing between members of a species (fitness = fitness/pop of species)
        self.is_speciated = True
        self.species_select = 'weighted' #Selecting randomly from all genomes, biases towards higher fitness
        self.thread_count = 1
        self.device = 'cpu' 
        self.population = 150
        self.generations = 100
        self.gens_to_improve = 50 #Gens for a species to improve before it is removed
        self.reenable_chance = 0.25 #Chance for a connection to be re-enabled during crossover
        self.species_c1       =  1.0 #params for species-distance algorithm
        self.species_c2       =  1.0
        self.species_c3       =  0.1
        self.max_species_dist =  1.5 #distance for genomes to be considered different species
        self.interspecies_crossover = 0.001
        self.mutate_ratio = 0.25 #percent of offspring made from mutation
        self.mutate_effect = 0.5 #Size of the mutations that occur; uniform random with this range centered on zero
        self.mutate_odds = [0.8, 0.9, 0.03, 0.05] #odds for mutating weights, perturbing values, add nodes, and adding connections
        self.elite_per_species   = 1 #Number of elite copied unchanged from each species
        self.elite_threshold   = 5 #Min size of species to get an elite copied over
        self.elite_range   = 3 #number of genomes checked for elite copy over (should be <= threshold, always)
        self.elite_evals   = 5 #Evaluations done on each genome in the elite range to find a more accurate fitness value
        self.activation_func = nn.Sigmoid()
        self.activation_const = 4.9
        self.NODE_INNOVATION_NUMBER = -1
        self.WEIGHT_INNOVATION_NUMBER = -1
        
#Cart pole config, pretty much solves the problem
#might not be optimal, but final solution about 200 (max score) each time
cart_pole = Experiment('CartPole')
cart_pole.env = gym.make('CartPole-v0')
cart_pole.device        = 'cpu'
cart_pole.inputs        = 4
cart_pole.outputs       = 2
cart_pole.layers        = 4 #seems that only 2 are necessary?
cart_pole.layer_size    = 4
cart_pole.trials        = 10
cart_pole.population    = 101
cart_pole.generations   = 1 #20    #needs 35 for stable no-oscillation
cart_pole.mutate_effect = 1.0/cart_pole.inputs
cart_pole.genome_file   = './Pickled Genomes/cart_genes.pjar' #'/Pickled Genomes/cart_genes.pjar'
cart_pole.thread_count  = 8
#---------------------------
#Frostbite
frostbite_1 = Experiment('Frostbite')
frostbite_1.env             = gym.make('Frostbite-ram-v0', frameskip=4)
frostbite_1.device          = 'cpu'
frostbite_1.inputs          = 128
frostbite_1.outputs         = 18
frostbite_1.layers          = 2
frostbite_1.layer_size      = 512
frostbite_1.trials          = 1
frostbite_1.population      = 251
frostbite_1.generations     = 500
frostbite_1.child_count     = 0
frostbite_1.mutate_range    = 20
frostbite_1.mutate_count    = frostbite_1.population - 1
frostbite_1.mutate_effect   = 0.002
frostbite_1.elite_count     = frostbite_1.population - (frostbite_1.child_count + frostbite_1.mutate_count)
frostbite_1.elite_range     = 10
frostbite_1.elite_evals     = 30
frostbite_1.outfile         = 'terminal' #"frostbite.txt"
frostbite_1.genome_file     = None
frostbite_1.thread_count    = 24
frostbite_1.max_frames      = 1000000000
#---------------------------
#Venture
venture_1 = Experiment('Venture')
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
cart_NEAT = NEATExperiment('CartPole_NEAT')
cart_NEAT.env         = gym.make('CartPole-v0')
cart_NEAT.device      = 'cpu'
cart_NEAT.inputs      = 4
cart_NEAT.outputs     = 2
cart_NEAT.trials      = 10
cart_NEAT.population  = 50
cart_NEAT.generations = 5
cart_NEAT.elite_range = 1
cart_NEAT.elite_evals = 1
cart_NEAT.thread_count = 1
#---------------------------
cart_multithread = Experiment('CartPole_mt')
cart_multithread.env = gym.make('CartPole-v0')
cart_multithread.device        = 'cpu'
cart_multithread.inputs        = 4
cart_multithread.outputs       = 2
cart_multithread.layers        = 4 #seems that only 2 are necessary?
cart_multithread.layer_size    = 4
cart_multithread.trials        = 10
cart_multithread.population    = 101
cart_multithread.generations   = 10    #needs 35 for stable no-oscillation
cart_multithread.child_count   = 0     #Experiment is basic GA, so no crossover
cart_multithread.mutate_range  = 10
cart_multithread.mutate_count  = cart_multithread.population - 1
cart_multithread.mutate_effect = 10/4
cart_multithread.elite_count   = cart_multithread.population - (cart_multithread.child_count + cart_multithread.mutate_count)
cart_multithread.elite_range   = 10
cart_multithread.elite_evals   = 1  
cart_multithread.thread_count  = 2
#---------------------------
frost_NEAT = NEATExperiment('Frostbite_NEAT')
frost_NEAT.env         = gym.make('Frostbite-ram-v0', frameskip=4)
frost_NEAT.device      = 'cpu'
frost_NEAT.inputs      = 128
frost_NEAT.outputs     = 18
frost_NEAT.trials      = 1
frost_NEAT.population  = 101
frost_NEAT.generations = 250
frost_NEAT.elite_range = 1
frost_NEAT.elite_evals = 3
frost_NEAT.thread_count = 24
frost_NEAT.species_c1   = 5.
frost_NEAT.species_c2       = 5.
frost_NEAT.species_c3       = 0.01
frost_NEAT.max_species_dist = 13.0
frost_NEAT.mutate_odds      = [0.8, 0.9, 0.06, 0.1]
frost_NEAT.max_frames       = 1000000000
#---------------------------
xor = NEATExperiment("XOR")
xor.env = XOR_env()
xor.inputs        = 2
xor.outputs       = 1
xor.trials        = 1
xor.population    = 150
xor.generations   = 500
xor.elite_range   = 1
xor.elite_evals   = 1 #Deterministic, so no need for multiple evals
#---------------------------
list = [cart_pole, frostbite_1, venture_1, cart_NEAT, cart_multithread, frost_NEAT, xor]


