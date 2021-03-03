import evolve_basic
import evolve_multithreaded
import pickle
import torch
import experiments
import sys

SELECT_COUNT = 20 #Number of genomes selected from elite 
SELECT_TRIALS = 10 #how many 

#Runs an experiment given as a command line args
#Mostly just a wrapper file to call one of the evolve files
if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor) #Could test with floats later for better speed
    experiment = None
    #read sys args for experiment to use; inputs correspond to names given in experiments.py
    if len(sys.argv) > 1:
        found = False
        for e in experiments.list:
            if e.name == sys.argv[1]:
                experiment = e
                found = True
        if not found:
            print("Requested experiment not found")
            assert False
    else:
        print("Missing argument for experiment to be run")
        assert False
    fit_pop = None
    #These may be joined eventually (multithreaded algorithm should run fine with 1 thread) but basic evolve is better for testing things
    if experiment.thread_count > 1:
        print("multithreading!")
        fit_pop, saved = evolve_multithreaded.evolve(experiment)
    else:
        fit_pop, saved = evolve_basic.evolve(experiment)
    #Currently gives a brief report and demonstration of the fittest individual
    if experiment.genome_file:
        file = open(experiment.genome_file, 'wb')
        pickle.dump(saved, file) #This saves those fitness scores as well; could fix this
    print()
    print("#-------------------------------#")
    print("Experiment has concluded normally")
    fittest_genome = max(saved, key=lambda g: g[1])[0]#Finds the fittest genome based on the fitness saved for it during elite trial evals; ignores fitness sharing
    fitness = fittest_genome.evalFitness(iters=200)
    #fittest = fit_pop.fittest()
    print("Highest Fitness:", fitness)
    input("Press enter to continue to animation")
    fittest_genome.experiment.trials = 1
    fittest_genome.evalFitness(True)
    fittest_genome.printToTerminal()
    
