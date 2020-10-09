#Defines the experiment class, which holds params for experiments that are accessed by the GA
#Also creates experiment instances and assigns the appropriate values to them

class Experiment():
    
    def __init__(self, name):
        self.name = name
        
#Cart pole config, pretty much solves the problem
#might not be optimal, but final solution about 200 (max score) each time
cart_pole = Experiment('CartPole-v0')
cart_pole.inputs        = 4
cart_pole.outputs       = 2
cart_pole.layers        = 4
cart_pole.layer_size    = 4
cart_pole.trials        = 10
cart_pole.population    = 100 
cart_pole.generations   = 10    #needs 35 for stable no-oscillation
cart_pole.select_range  = 10
cart_pole.mutate_rate   = 10