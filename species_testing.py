#Testing file to run species distance algorithm on new populations; helpful in setting parameters

import experiments
import copy
import genome
from genome_NEAT import *

experiment = experiments.frost_NEAT

#Creates new population, based on evolve algorithm
pop_size = experiment.population
pop = []
for i in range(pop_size):
    new_net = "Maybe I can write a function to make a new net of type specified by the experiment"
    if experiment.genome == 'NEAT':
        new_net = NEATGenome(experiment)
    else:
        new_net = Genome(experiment)
    pop.append(new_net)
    
print("population created")
    
experiment.species_c1 = 1.0
experiment.species_c2 = 0.0
experiment.species_c3 = 0.0
genomes = copy.deepcopy(pop)
dist = 0
count = 0
for i in range(pop_size):
    genome = genomes[0]
    for j in range(pop_size-i-1):
        dist += genome.speciesDistance(genomes[j+1])
        count += 1
    del genomes [0]
avg_1 = dist/count
print("Average first value is:", avg_1)

experiment.species_c1 = 0.0
experiment.species_c2 = 1.0
experiment.species_c3 = 0.0
genomes = copy.deepcopy(pop)
dist = 0
count = 0
for i in range(pop_size):
    genome = genomes[0]
    for j in range(pop_size-i-1):
        dist += genome.speciesDistance(genomes[j+1])
        count += 1
    del genomes [0]
avg_2 = dist/count
print("Average second value is:", avg_2)

experiment.species_c1 = 0.0
experiment.species_c2 = 0.0
experiment.species_c3 = 1.0
genomes = copy.deepcopy(pop)
dist = 0
count = 0
for i in range(pop_size):
    genome = genomes[0]
    for j in range(pop_size-i-1):
        dist += genome.speciesDistance(genomes[j+1])
        count += 1
    del genomes [0]
avg_3 = dist/count
print("Average third value is:", avg_3)
