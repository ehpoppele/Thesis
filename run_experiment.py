import evolve_basic
import evolve_multithreaded
import torch
import experiments
import sys

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
        fit_pop = evolve_multithreaded.evolve(experiment)
    else:
        fit_pop = evolve_basic.evolve(experiment)
    #Currently gives a brief report and demonstration of the fittest individual
    print()
    print("#-------------------------------#")
    print("Experiment has concluded normally")
    fittest = fit_pop.fittest()
    print("Highest Fitness:", fit_pop.fittest().fitness)
    input("Press enter to continue to animation")
    fittest.experiment.trials = 1
    fittest.evalFitness(True)
    fittest.printToTerminal()
    
