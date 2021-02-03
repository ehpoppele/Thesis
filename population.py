import random
import math
import multiprocessing

#Population class is a sorted list to hold genomes
#Doesn't use python queue since I also need to index it
class Population():

    def __init__(self, experiment):
        self.experiment = experiment
        self.genomes = []
        self.species = []
        self.lock = multiprocessing.Lock() #Is this still needed?
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
                    genome.species = species
                    assigned = True
                    break
            if not assigned:
                new_species = Species(genome, self.experiment)
                genome.species = new_species
                self.species.append(new_species)
        
    #Returns a genome at random from the n fittest genomes in the population
    #In the case of NEAT, it returns from a random pool of all the top genes of each species
    #With each gene coming from the top (n/pop size) percentile of its species
    #Might end up with more than n genes to choose from 
    def fittest(self, search_range):
        if self.is_speciated:
            top_genes = []
            percentile = search_range/self.size()
            for species in self.species:
                target = math.ceil(percentile*species.size())
                for i in range(target):
                    top_genes.append(species[i])
            return random.choice(top_genes)
        else:
            return self.genomes[random.randint(0, range-1)]
            
    #Returns the highest fitness individual in the population
    def top_fittest(self):
        return self.genomes[0]
        
    #Returns a set containing the <range> top fittest individuals 
    #Uses the same rounding method as above, but needs to return exactly fit_range number of individuals,
    #So this may drop some of the later species; might be a problem later so I'll look into fixing it
    def fitSet(self, fit_range):
        if self.is_speciated:
            fit_set = set()
            size = 0
            percentile = fit_range/len(self.genomes)
            for species in self.species:
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
        
    #Update species if they haven't been improved in long enough (setting all genomes' fitness to zero)
    #Also removes empty species
    def checkSpeciesForImprovement(self, gens_to_improve):
        to_remove = []
        for species in self.species:
            if species.size() <= 0:
                to_remove.append(species)
            else:
                species.checkForImprovement(gens_to_improve)
        #Slow but shouldn't happen too often
        for species in to_remove:
            self.species.remove(species)
        all_expired = True
        for s in self.species:
            if s.can_reproduce:
                all_expired = False
                break
        if all_expired:
            print("No species have improved recently; fix this later.")
            assert False
            
    def assignOffspringProportions(self):
        total_fitness = 0
        for s in self.species:
            if s.can_reproduce:
                total_fitness += s.sumFitness()
        for s in self.species:
            if s.can_reproduce:
                s.offspring_proportion = s.sumFitness()/total_fitness
        
    
class Species():

    def __init__(self, representative, experiment, add_rep=True):
        self.genomes = []
        self.experiment = experiment
        self.selection_type = experiment.species_select
        self.lock = multiprocessing.Lock() #??
        self.rep = representative
        self.gens_since_improvement = 0
        self.can_reproduce = True
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
                self.can_reproduce = False

    #returns the total sum of fitness of all genomes in the species
    def sumFitness(self):
        sum = 0
        for g in self.genomes:
            sum += g.fitness
        return sum
        
    #Selects an individual from the population based on its fitness; may have different behavior for different algorithms    
    def select(self):
        if self.selection_type == "range":
            return self.fittest(self.experiment.mutate_range)
        if self.selection_type == "weighted":
            selection = random.uniform(0, self.sumFitness())
            for g in self.genomes:
                selection -= g.fitness
                if selection <= 0:
                    return g
            #If we miss them all by some rounding error
            return self.genomes[-1]
        #Default return, though I don't need this since this isn't C
        return self.genomes[0]
            







   