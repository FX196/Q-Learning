import tensorflow as tf
import numpy as np
import random
from collections import deque

gamma = 0.95        # discount factor
epsilon_0 = 0.01    # epsilon-greedy
minibatch_size = 40
replay_memory_size = 1000000

class DQN:
    def __init__(self, action_space):
        self.replay_memory = deque()
        self.time_step = 0
        self.epsilon = epsilon_0
        self.action_space = action_space

