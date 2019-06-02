import random

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten

# hyperparameters
observe_time = 1000
max_memory_size = 20000
epsilon = 0.95
gamma = 0.9
mb_size = 100


class Model:
    def __init__(self, action_space):
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=(210, 160, 3), filters=32, kernel_size=(8, 8), strides=4, activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"))
        self.model.add(Flatten())
        self.model.add(Dense(units=256, activation="relu"))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(units=action_space, activation="linear"))
        self.model.compile(optimizer="adam", loss="mse")
        self.state_size = None
        self.action_space = action_space
        self.memory = []
        self.episode = 0
        self.epsilon = 0.95

    def get_action(self, observation, verbose=False):
        prediction = self.model.predict(observation)
        if verbose:
            print(prediction)
        return np.argmax(prediction)

    def observe(self, env, with_graphics=False):
        observation = env.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs, obs), axis=1)
        done = False

        # observe for a set amount of timesteps and add the observations to memory
        # uses epsilon-greedy with epsilon annealed over time
        for t in range(observe_time):
            if np.random.rand() <= self.epsilon:
                action = env.action_space.sample()
            else:
                Q = self.model.predict(state)
                action = np.argmax(Q)
            observation_new, reward, done, info = env.step(action)
            obs_new = np.expand_dims(observation_new, axis=0)
            state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)
            self.memory.append((state, action, reward, state_new, done))
            state = state_new
            if done:
                env.reset()
                obs = np.expand_dims(observation, axis=0)
                state = np.stack((obs, obs), axis=1)

        self.epsilon *= 0.9
        # finish observation

    def train(self):
        minibatch = random.sample(self.memory, mb_size)
        inputs_shape = (mb_size,) + minibatch[0][0].shape[1:]
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((mb_size, self.action_space))

        for i in range(mb_size):
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            state_new = minibatch[i][3]
            done = minibatch[i][4]

            inputs[i:i + 1] = np.expand_dims(state, axis=0)
            targets[i] = self.model.predict(state)
            Q_sa = self.model.predict(state_new)

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + gamma * np.max(Q_sa)

            loss = self.model.train_on_batch(inputs, targets)
            print("\rEpisode = %s,Loss = %.5f" % (str(self.episode), loss))

        if self.memory >= max_memory_size:
            self.memory = self.memory[-max_memory_size:]

    def play(self, env):
        observation = env.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs, obs), axis=1)
        done = False
        tot_reward = 0.0

        while not done:
            env.render()  # Uncomment to see game running
            Q = self.model.predict(state)
            action = np.argmax(Q)
            observation, reward, done, info = env.step(action)
            obs = np.expand_dims(observation, axis=0)
            state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
            tot_reward += reward
        print('Game ended! Total reward: {}'.format(tot_reward))
