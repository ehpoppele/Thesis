# Thesis
Senior Undergraduate Thesis Investigating Neuroevolution
Completed for the Reed College Computer Science Department, 2021

The project is accompanied by a written thesis, which will be included or linked once it is finished. 

This readme is a work in progress. It is not in a completed state, and will be sometime after I actually finish up the written portion.

## Neuroevolution
This project implements the NEAT algorithm for neuroevolution (http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), as well as another algorithm from Uber's AI division (https://arxiv.org/pdf/1712.06567.pdf). A third type of genome, appearing in genome_Tensor.py, contains my own improvements to the NEAT algorithm to make it more applicable to this specific problem domain (playing Atari games).

## Installation
This implementation uses pytorch for neural networks and OpenAI's Gym environment for problems that the networks are run on, including Atari games which are implemented in Gym with the Arcade Learning Environment (ALE).

Installing gym can be done with `pip install gym` and `pip install gym[atari]` for the Atari dependencies that most of the experiments use. Pytorch installation instructions vary depending on OS and package manager, and can be found at https://pytorch.org/. The CUDA option for pytorch is needed to run any networks on the GPUs; this can be set as an option for the experiments in my code, but networks are evaluated on the CPU by default.

## Running Experiments
Experiments are defined as a class in the `experiments.py` file, which specifies all parameters, including the genome type and atari game or other environment to train the networks on. `run_experiment.py`, the main program, currently takes a single command line argument specifying the pre-configured experiment to be run. The options for this argument are given by the last line of `experiment.py`; the two most stable are currently `Frostbite` and `CartPole`. Other arguments to specify frame limit, genome type, and other parameters will be added eventually and described in more detail here; in the meantime modifications can be made by changing one of the existing experiment objects or creating a new one.

`run_experiment.py` will output directly to the terminal and can take up to 14 hours (with 24 threads) to complete, depending on the selecting experiment. It may be useful to use `nohup` or another utility to manage this and track the output in a text file, such as with `nohup python3 run_experiment.py Frostbite > frostbite_results.txt`.

## Known Errors and Issues
When multithreading, if the program crashes with a "too many open files" error, trying raising the user file limit (`ulimit -n <value>` on Linux) or reducing the population size. Due to the way python multiprocessing works, the file limit must be nearly 20 times the population size, in my experience, to prevent this error.

CUDA does not work very well with the program; I would like to blame this on python multiprocessing again, though it could be my own fault. The multiprocessing launches a new instance of CUDA on the GPU for each thread, the overhead of which quickly fills the available memory. With a low thread count, using the GPU may still accelerate the program rather than crashing it, but otherwise it requires a significant workaround that I am not aware of.

Every file in the `TestFiles` directory worked at some point, but it is likely that none of them work now, as most are for older versions of the main program files.

