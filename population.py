#Defines the Species and Population classes, which track and organize the genomes
#The population class builds on the species class, and so species is defined first

import random
import math
import multiprocessing 
    
#Tracks all genomes that are of the same species, sorted by fitness
class Species():

    #Defined with a representative to which other genomes are compared to see if they fall in the species
    def __init__(self, experiment, representative, add_rep=True, gens_since_improvement=0, last_fittest=0, can_reproduce=True):
        self.genomes = []
        self.experiment = experiment
        self.selection_type = experiment.species_select #How fit genomes are selected for mutation/crossover
        if self.selection_type == "range":
            self.select_range = experiment.mutate_range #This may need to change later
        #self.lock = multiprocessing.Lock() #will review if this is needed once I fix MT
        self.rep = representative
        self.gens_since_improvement = 0
        self.can_reproduce = True
        self.last_fittest = -1 #Assumes fitness scores won't be negative
        if add_rep:
            self.add(representative)
        #If not, then we are copying over an old species, so we retain the same data
        else:
            self.gens_since_improvement = gens_since_improvement
            self.last_fittest = last_fittest
            self.can_reproduce = can_reproduce
         
    def __getitem__(self, index):
        return self.genomes[index]
        
    def size(self):
        return len(self.genomes)
        
    #Returns the highest fitness individual in the population
    def fittest(self):
        return self.genomes[0]
        
    #returns the total sum of fitness of all genomes in the species
    def sumFitness(self):
        sum = 0
        for g in self.genomes:
            sum += g.fitness
        return sum
        
    #Selects an individual from the population based on its fitness; may have different behavior for different algorithms    
    def select(self):
        if self.selection_type == "range":
            return self.genomes[random.randint(0, self.select_range)]
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
     
    #Returns a random genome from this species (for selecting species reps in a new generation)
    def randOfSpecies(self):
        return random.choice(self.genomes)
    
    #Sorts the population by fitness again, to be used after mass changes to fitness (like with fitness sharing)
    def reorder(self): #use python library sort
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
    def checkSpeciesForImprovement(self, gen_limit):
        if self.genomes[0].fitness > self.last_fittest:
            self.gens_since_improvement = 0
            self.last_fittest = self.genomes[0].fitness
        else:
            self.gens_since_improvement += 1
            if self.gens_since_improvement >= gen_limit:
                self.can_reproduce = False


#The population tracks all genomes in one array but also tracks each individual species
class Population(Species):

    def __init__(self, experiment):
        self.experiment = experiment
        self.selection_type = experiment.species_select
        self.genomes = []
        self.species = []
        #self.lock = multiprocessing.Lock() #Is this still needed?
        self.species_num = 0
        self.is_speciated = True
        
    #Works as above, adding in sorted position to genome list
    #And also adds genome to the appropriate species, or creates a new one when needed
    def add(self, genome):
        super().add(genome)
        if self.is_speciated:
            assigned = False
            for species in self.species:
                if genome.speciesDistance(species.rep) < genome.experiment.max_species_dist:
                    species.add(genome)
                    assigned = True
                    break
            #Genome becomes the rep for a new species
            if not assigned:
                new_species = Species(self.experiment, genome)
                self.species.append(new_species)
    
    #Sorts the population by fitness again, to be used after mass changes to fitness (like with fitness sharing)
    #Using insertion sort since the list should be close to sorted already
    def reorder(self):
        for species in self.species:
            species.reorder()
        super().reorder()
    
    #Used to select a random representative of the species for the next generation
    def randOfSpecies(self, species):
        for s in self.species:
            if s == species:
                return s.randOfSpecies()
        print("Looking for a species that could not be found")
        assert False #Not sure how to handle this error yet (which should never happen) so we just crash it
        
    #Update species if they haven't been improved in long enough (setting all genomes' fitness to zero)
    #Also removes empty species
    def checkSpeciesForImprovement(self, gens_to_improve):
        to_remove = []
        for species in self.species:
            if species.size() <= 0:
                to_remove.append(species)
            else:
                species.checkSpeciesForImprovement(gens_to_improve)
        #Slow but shouldn't happen too often
        for species in to_remove:
            self.species.remove(species)
        all_expired = True
        for s in self.species:
            if s.can_reproduce:
                all_expired = False
                break
        if all_expired:
            #Repopulate from top two species
            highest_avg_fitness = -1
            top = None
            second = None
            for s in self.species:
                avg = s.sumFitness()/s.size()
                if avg >= highest_avg_fitness:
                    highest_avg_fitness = avg
                    top = s
                    second = top
                top.can_reproduce = True #This should never be None so we're okay with crashing if it gets there
                if second is not None:
                    second.can_reproduce = True
            
    #Assigns a percentage to each species to track how many offspring they will get for the next generation
    #This is based on the total sum of the species fitness
    #If fitness sharing is used, it is effectively based on the average fitness
    def assignOffspringProportions(self):
        total_fitness = 0
        for s in self.species:
            if s.can_reproduce:
                total_fitness += s.sumFitness()
        if total_fitness == 0:
            total_fitness = 1 #Avoid division by zero
        for s in self.species:
            if s.can_reproduce:
                if total_fitness == 0:
                    s.offspring_proportion = s.size()/self.size()
                else:
                    s.offspring_proportion = s.sumFitness()/total_fitness
            else:
                s.offspring_proportion = 0
    
    






   
