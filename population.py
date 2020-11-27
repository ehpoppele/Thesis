import random
import multiprocessing

#Population class is a sorted list to hold genomes
#Doesn't use python queue since I also need to index it
class Population():

    def __init__(self):
        self.genomes = []
        self.lock = multiprocessing.Lock()
        
    #Adds to the population queue, maintaining order
    def add(self, genome):
        added = False
        for j in range(len(self.genomes)):
            if genome.fitness > self.genomes[j].fitness:
                self.genomes.insert(j, genome)
                added = True
                break
        if not added:
            self.genomes.append(genome)
        
    #Returns a genome at random from the n fittest genomes in the population
    def fittest(self, n):
        return self.genomes[random.randint(0, n-1)]
        
    def __getitem__(self, index):
        return self.genomes[index]
        
    def size(self):
        return len(self.genomes)
        
    #Returns a set containing the <range> top fittest individuals 
    def fitSet(self, fit_range):
        assert fit_range < self.size(), "Tried to get the " + str(range) + "fittest individuals in a population, but it only has " + str(self.size) + "total genomes."
        ret = set()
        for i in range(fit_range):
            ret.add(self.genomes[i])
        return ret