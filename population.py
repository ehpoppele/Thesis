import random
import math
import multiprocessing 
    
class Species():

    def __init__(self, experiment, representative, add_rep=True):
        self.genomes = []
        self.experiment = experiment
        self.selection_type = experiment.species_select
        self.select_range = experiment.mutate_range #This may need to change later
        self.lock = multiprocessing.Lock() #??
        self.rep = representative
        self.gens_since_improvement = 0
        self.can_reproduce = True
        self.last_fittest = -1 #Assumes fitness scores won't be negative
        if add_rep:
            self.add(representative)
        #If not, then we are copying over an old species, so we retain the same data
        else:
            self.gens_since_improvement = representative.species.gens_since_improvement
            self.last_fittest = representative.species.last_fittest
            self.can_reproduce = representative.species.can_reproduce
         
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
            return self.genomes[random.randint(0, self.select range)]
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
     
    #Returns a random genome from this species (for selecting species reps etc)
    def randOfSpecies(self):
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
    def checkSpeciesForImprovement(self, gen_limit):
        if self.genomes[0].fitness > self.last_fittest:
            self.gens_since_improvement = 0
            self.last_fittest = self.genomes[0].fitness
        else:
            self.gens_since_improvement += 1
            if self.gens_since_improvement >= gen_limit:
                self.can_reproduce = False


            
class Population(Species):

    def __init__(self, experiment):
        self.experiment = experiment
        self.selection_type = experiment.species_select
        self.genomes = []
        self.species = []
        self.lock = multiprocessing.Lock() #Is this still needed?
        self.species_num = 0
        self.is_speciated = False
        
    #Works as above, adding in sorted position to genome list
    #And also adds genome to the appropriate species, or creates a new one when needed
    def add(self, genome):
        super().add(genome)
        if self.is_speciated:
            assigned = False
            for species in self.species:
                if genome.speciesDistance(species.rep) < genome.experiment.max_species_dist:
                    species.add(genome)
                    genome.species = species
                    assigned = True
                    break
            #Genome becomes the rep for a new species
            if not assigned:
                new_species = Species(self.experiment, genome)
                genome.species = new_species
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
            print("No species have improved recently; fix this later.") #NEAT paper at this point refocuses into top 2 species I believe; I'll code that up if I ever see this happen
            assert False
            
    def assignOffspringProportions(self):
        total_fitness = 0
        for s in self.species:
            if s.can_reproduce:
                total_fitness += s.sumFitness()
        for s in self.species:
            if s.can_reproduce:
                s.offspring_proportion = s.sumFitness()/total_fitness
            else:
                s.offspring_proportion = 0
    
    






   