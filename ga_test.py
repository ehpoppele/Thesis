import basic_evolve
import torch
import gym
import random
import experiments

#experiment = {"name" : 'CartPole-v0', "inputs" : 4, "outputs" : 2, "layers" : 3, "layer_size" : 16, "trials" : 20}
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    experiment = experiments.cart_pole
    fit_pop = basic_evolve.evolve(experiment.population, experiment.generations, experiment, experiment.select_range)
    print("fittest:", fit_pop[0].fitness)
    fittest = fit_pop[0]
    env = gym.make(experiment.name)
    observation = env.reset()
    sum_reward = 0
    input("press enter to continue to animation")
    #render not working?
    for t in range(200):
        env.render()
        inputs = torch.from_numpy(observation)
        outputs = fittest.model(inputs)
        action = 0
        rand_select = random.random()
        for i in range(len(outputs)):
            rand_select -= outputs[i]
            if rand_select < 0:
                action = i
                break
        observation, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            break
    env.close()
    fittest.fitness = sum_reward
    print(sum_reward)