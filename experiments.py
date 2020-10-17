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
cart_pole.layers        = 4 #seems that only 2 are necessary?
cart_pole.layer_size    = 4
cart_pole.trials        = 10
cart_pole.population    = 100 
cart_pole.generations   = 10    #needs 35 for stable no-oscillation
cart_pole.child_count   = 0     #Experiment is basic GA, so no crossover
cart_pole.mutate_range  = 10
cart_pole.mutate_count  = cart_pole.population - 1
cart_pole.mutate_effect = 10
cart_pole.elite_count   = cart_pole.population - (cart_pole.child_count + cart_pole.mutate_count)
cart_pole.elite_range   = 10
cart_pole.elite_evals   = 30
#---------------------------
#Frostbite
frostbite_1 = Experiment('Frostbite-ram-v0')
frostbite_1.inputs          = 128
frostbite_1.outputs         = 18
frostbite_1.layers          = 1
frostbite_1.layer_size      = 256
frostbite_1.trials          = 3
frostbite_1.population      = 100
frostbite_1.generations     = 8
frostbite_1.child_count     = 0
frostbite_1.mutate_range    = frostbite_1.population // 10
frostbite_1.mutate_count    = frostbite_1.population - 1
frostbite_1.mutate_effect   = 1
frostbite_1.elite_count     = frostbite_1.population - (frostbite_1.child_count + frostbite_1.mutate_count)
frostbite_1.elite_range     = frostbite_1.mutate_range
frostbite_1.elite_evals     = 30