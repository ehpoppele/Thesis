#Main file for basic GA program
#camelCaps is for func names, snake_case is variables/objects

#Evolve currently takes a whole set of params and returns the final population after the experiment concludes
def evolve(pop_size, generation_count, experiment, selection_range):#Add a metric boatload of params in here
    #Create new random population, sort by starting fitness
    population = []
    for i in range(pop_size):
        new_net = Genome(experiment) #experiment will give info on inputs, outputs, layer size and count via a config. Initial random for some range
        new_net.evalFitness(experiment) #Fitness is data member of genome; game is arg to identify game being played. Maybe passed func?
        #Maybe have fitness auto-evaled when new genome is made? maybe not
        #Add new genome to the population, keeping population sorted by fitness
        #If population becomes a class, this would be moved into a method for it (pop push something etc)
        added = False
        for j in range(len(population)):
            if new_net.fitness > population[j].fitness:
                population.insert(j, new_net)
                added = True
                break
        if not added:
            population.append(new_net)
    for g in range(generation_count):
        #Make new population from mutations plus best net from last pop
        new_pop = []
        for i in range(popSize - 1):
            parent = population[random in range selection_range]#---------------------------------fix this!----------------------------------
            new_net = parent.mutate()#Don't know what params will go here yet
            new_net.evalFitness(experiment)
            #same loop before to add to sorted pop
            added = False
            for j in range(len(new_pop)):
                if new_net.fitness > new_pop[j].fitness:
                    new_pop.insert(j, new_net)
                    added = True
                    break
            if not added:
                new_pop.append(new_net)
        #Find top individual from current pop to carry over
        #Do so by retesting each of the top ten 30 times and taking the average
        best_fitness = -inf
        fittest = None
        for i in range(10):
            fitsum = 0
            for j in range(30):
                fitsum += population[i].evalFitness(experiment) #eval will also return the new fitness, not just update it
            if fitsum/30 > best_fitness:
                best_fitness = fitsum/30
                fittest = population[i]
        new_pop.append(fittest)
        population = new_pop
    return population
    
    
"""
Needed before this can run:
    genome evalFitness
    main program file to run this, do some visual output on fittest individual, maybe gather some data

"""
                
        