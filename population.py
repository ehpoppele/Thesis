import random
import math
import multiprocessing

#Population class is a sorted list to hold genomes
#Doesn't use python queue since I also need to index it
class Population():

    def __init__(self):
        self.genomes = []
        self.lock = multiprocessing.Lock()
        self.species_memo = []
        self.species_num = 0
        self.NEATGenes = False
        
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
    #In the case of NEAT, it returns from a random pool of all the top genes of each species
    #With each gene coming from the top (n/pop size) percentile of its species
    #Might end up with more than n genes to choose from 
    def fittest(self, n):
        if self.NEATGenes:
            if self.species_memo == []:
                self.makeSpeciesMemo()
            top_genes = []
            percentile = n/len(self.genomes)
            for species in self.species memo:
                target = math.ceil(percentile*len(species))
                for i in range(target):
                    top_genes.append(species[i])
            return random.choice(top_genes)
        else:
            return self.genomes[random.randint(0, n-1)]
        
    def __getitem__(self, index):
        return self.genomes[index]
        
    def size(self):
        return len(self.genomes)
        
    #Returns a set containing the <range> top fittest individuals 
    #Uses the same rounding method as above, but needs to return exactly fit_range number of individuals,
    #So this may drop some of the later species; might be a problem later so I'll look into fixing it
    def fitSet(self, fit_range):
        if self.NEATGenes:
            if self.species_memo == []:
                self.makeSpeciesMemo()
            fit_set = set()
            size = 0
            percentile = n/len(self.genomes)
            for species in self.species memo:
                target = math.ceil(percentile*len(species))
                for i in range(target):
                    if size < fit_range:
                        fit_set.append(species[i])
                        size += 1
                    else:
                        break
            return fit_set
        else:
            assert fit_range < self.size(), "Tried to get the " + str(range) + "fittest individuals in a population, but it only has " + str(self.size) + "total genomes."
            ret = set()
            for i in range(fit_range):
                ret.add(self.genomes[i])
            return ret
            
    #This makes separate lists for each species
    #Since the main array of genomes is sorted, these should be as well
    def makeSpeciesMemo(self):
        species_max = max[g.species for g in self.genomes]
            for _ in range(species_max+1):
                self.species_memo.append([])
            for g in self.genomes:
                self.species_memo[g.species].append(g)
     
    #Returns a random genome of the given species. Memoizes the current population with their species if it has not already been done
    def randOfSpecies(self, species_num):
        if self.species_memo == []:
            self.makeSpeciesMemo()
        return random.choice(self.species_memo[species_num])








   