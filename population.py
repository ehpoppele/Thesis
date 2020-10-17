import random

#Population class is a sorted list to hold genomes
#Doesn't use python queue since I also need to index it
class Population():

    def __init__(self, size):
        self.genomes = []
        self.total_size = size #size of full starting population; not always equal to len of genomes (as one can see here)
        
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
        





#needs:
#queue for holding genomes, sorted by fitness
#