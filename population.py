import random
import math
import multiprocessing

#Population class is a sorted list to hold genomes
#Doesn't use python queue since I also need to index it
class Population():

    def __init__(self):
        self.genomes = []
        self.species = []
        self.lock = multiprocessing.Lock() #Is this still needed?
        self.species_memo = []
        self.species_num = 0
        self.is_speciated = False
        
    def __getitem__(self, index):
        return self.genomes[index]
        
    def size(self):
        return len(self.genomes)
        
    #Adds to the population queue, maintaining order
    #Also assigns a species and adds to the species if population is speciated
    def add(self, genome):
        added = False
        for j in range(len(self.genomes)):
            if genome.fitness > self.genomes[j].fitness:
                self.genomes.insert(j, genome)
                added = True
                break
        if not added:
            self.genomes.append(genome)
        if self.is_speciated:
            assigned = False
            for species in self.species:
                if genome.speciesDistance(species.rep) < genome.experiment.max_species_dist:
                    species.add(genome)
                    g.species = species
                    assigned = True
                    break
            if not assigned:
                new_species = Species(genome)
                genome.species = new_species
        
    #Returns a genome at random from the n fittest genomes in the population
    #In the case of NEAT, it returns from a random pool of all the top genes of each species
    #With each gene coming from the top (n/pop size) percentile of its species
    #Might end up with more than n genes to choose from 
    def fittest(self, range):
        if self.is_speciated:
            top_genes = []
            percentile = range/self.size()
            for species in self.species:
                target = math.ceil(percentile*species.size())
                for i in range(target):
                    top_genes.append(species[i])
            return random.choice(top_genes)
        else:
            return self.genomes[random.randint(0, range-1)]
        
        
    #Returns a set containing the <range> top fittest individuals 
    #Uses the same rounding method as above, but needs to return exactly fit_range number of individuals,
    #So this may drop some of the later species; might be a problem later so I'll look into fixing it
    def fitSet(self, fit_range):
        if self.speciated:
            fit_set = set()
            size = 0
            percentile = fit_range/len(self.genomes)
            for species in self.species_memo:
                target = math.ceil(percentile*species.size())
                for i in range(target):
                    if size < fit_range:
                        fit_set.add(species[i])
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
    
    #Sorts the population by fitness again, to be used after mass changes to fitness (like with fitness sharing)
    #Using insertion sort since the list should be close to sorted already
    def reorder(self):
        for species in self.species:
            species.reorder()
        #Could replace this with a merge sort drawing from sorted species
        #Should do that
        for i in range(self.size()):
            genome = self.genomes[i]
            j = i - 1
            while j > 0:
                if genome.fitness > self.genomes[j].fitness:
                    self.genomes[j+1] = self.genomes[j]
                    self.genomes[j] = genome
                else:
                    break
                j-=1
    
    def randOfSpecies(self, species):
        for s in self.species:
            if s == species:
                return s.randSelect()
        print("Looking for a species that could not be found")
        assert False
        return
    
class Species():

    def __init__(self, representative, add_rep=True):
        self.genomes = []
        self.lock = multiprocessing.Lock() #??
        self.rep = representative
        self.gens_since_improvement = 0
        self.last_fittest = -1 #Assumes fitness scores won't be negative
        if add_rep:
            self.add(representative)
         
    def __getitem__(self, index):
        return self.genomes[index]
        
    def size(self):
        return len(self.genomes)
        
    #Adds to the species queue, maintaining order
    def add(self, genome):
        added = False
        for j in range(len(self.genomes)):
            if genome.fitness > self.genomes[j].fitness:
                self.genomes.insert(j, genome)
                added = True
                break
        if not added:
            self.genomes.append(genome)
        
    #Returns a genome at random from the n fittest genomes in the species
    def fittest(self, n):
        return random.choice(self.genomes)
 
    #Returns a set containing the <range> top fittest individuals
    def fitSet(self, fit_range):
        assert fit_range < self.size(), "Tried to get the " + str(range) + "fittest individuals in a species, but it only has " + str(self.size()) + "total genomes."
        ret = set()
        for i in range(fit_range):
            ret.add(self.genomes[i])
        return ret
     
    #Returns a random genome from this species (for selecting species reps etc)
    def randSelect(self):
        return random.choice(self.genomes)
    
    #Sorts the population by fitness again, to be used after mass changes to fitness (like with fitness sharing)
    def reorder(self):
        for i in range(self.size()):
            genome = self.genomes[i]
            j = i - 1
            while j > 0:
                if genome.fitness > self.genomes[j].fitness:
                    self.genomes[j+1] = self.genomes[j]
                    self.genomes[j] = genome
                else:
                    break
                j-=1
    
    #Checks if the species has improved its max fitness and updates based on that
    #after the given number of generations, the species will be stopped if it has not improved
    #and all genomes will be set to zero fitness to avoid having them reproduce
    def checkForImprovement(self, gen_limit):
        if self.genomes[0].fitness > self.last_fittest:
            self.gens_since_improvement = 0
        else:
            self.gens_since_improvement+=1
            if self.gens_since_improvement >= gen_limit:
                for g in self.genomes:
                    g.fitness = 0








   