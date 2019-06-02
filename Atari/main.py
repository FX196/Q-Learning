import gym
import time
import numpy as np
import random
from Atari import model

env = gym.make("Breakout-v0")

epsilon = 1


if __name__ == "__main__":
    model = model.Model(4)

    for i_episode in range(20):
        observation = env.reset()
        print(observation.shape)
        for t in range(100):
            env.render()
            time.sleep(0.1)
            if random.random() > epsilon:
                action = model.get_action(np.expand_dims(observation, axis=0))
            else:
                action = np.random.randint(0, 4)
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()