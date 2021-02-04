#Test file for tinkering about in Open AI gym. Also useful to check input/output spaces of environments

import gym

env = gym.make('Frostbite-ram-v0')
for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(type(observation))
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()