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
        model.observe(env)
        model.train()
    print("Playing")
    model.play(env)