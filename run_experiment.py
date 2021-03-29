import evolve_basic
import evolve_multithreaded
import pickle
import torch
import experiments
import sys
import statistics
from torch.multiprocessing import set_start_method

SELECT_COUNT = 20 #Number of genomes selected from elite 
SELECT_TRIALS = 10 #how many 

#Runs an experiment given as a command line args
#Mostly just a wrapper file to call one of the evolve files
if __name__ == "__main__":
    trials = 1
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
    final_vals = []
    set_start_method('spawn')
    for i in range(trials):
        fit_pop = None
        #These may be joined eventually (multithreaded algorithm should run fine with 1 thread) but basic evolve is better for testing things
        if experiment.thread_count > 1:
            print("Multithreading! Iteration ", i)
            fit_pop, saved = evolve_multithreaded.evolve(experiment)
        else:
            fit_pop, saved = evolve_basic.evolve(experiment)
        #Currently gives a brief report and demonstration of the fittest individual
        fits = []
        fittest = max(saved, key=lambda g: g[1])#Finds the fittest genome based on the fitness saved for it during elite trial evals; ignores fitness sharing
        fittest_genome = fittest[0]
        if experiment.genome_file:
            file = open(experiment.genome_file + str(i) + ".pjar", 'wb')
            pickle.dump(fittest_genome, file) #This saves those fitness scores as well; could fix this
        for j in range(200):
            fits.append(fittest_genome.evalFitness())
        exp_vals = [fittest[1], statistics.mean(fits), statistics.median(fits), max(fits), min(fits), fittest_genome]
        print(exp_vals)
        final_vals.append(exp_vals)

    print()
    print("#-------------------------------#")
    print("All experiment have concluded normally")
    for i in range(trials):
        print("Fitness before many evals:", final_vals[i][0])
        print("Mean Score:", final_vals[i][1])
        print("Median Score:", final_vals[i][2])
        print("Highest Score:", final_vals[i][3])
        print("Lowest Score:", final_vals[i][4])
        #fittest_genome.evalFitness(True)
        final_vals[i][5].printToTerminal()
    
