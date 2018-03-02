# noinspection PyUnresolvedReferences
import random
import sys

import numpy as np
from game import *
from keras.layers import Dense, Flatten
from keras.models import Sequential

# hyperparameters
observe_time = 200
epsilon = 0.95
gamma = 0.9
mb_size = 50


def build_model():
    model = Sequential()
    model.add(Dense(20, input_shape=(2,)+(4,4), init='uniform', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, init='uniform', activation='relu'))
    model.add(Dense(128, init='uniform', activation='relu'))
    model.add(Dense(64, init='uniform', activation='relu'))
    model.add(Dense(16, init='uniform', activation='relu'))
    model.add(Dense(4, init='uniform', activation='relu'))
    model.compile(optimizer='adam', loss='mse')
    return model


def observe(env, model, D):
    # initialize observation
    observation = env.board
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs,obs), axis=1)
    done = False

    for t in range(observe_time):
        if np.random.rand() <= epsilon:
            action = random.randint(0,3)
        else:
            Q = model.predict(state)
            action = np.argmax(Q)
        observation_new, reward, done, legal = env.step(action, True)
        obs_new = np.expand_dims(observation_new, axis=0)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :, :], axis=1)
        D.append((state, action, reward, state_new, done))
        state = state_new
        if done:
            env = Game(4)
            observation = env.board
            obs = np.expand_dims(observation, axis=0)
            state = np.stack((obs, obs), axis=1)
            done = False
    return D

def train(env, model, D, episode):
    minibatch = random.sample(D, mb_size)

    inputs_shape = (mb_size,) + D[0][0].shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, 4))

    for i in range(mb_size):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]

        inputs[i:i+1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)

        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

        loss = model.train_on_batch(inputs, targets)
        sys.stdout.write("\rEpisode = %s, Loss = %.5f" %(str(episode), loss))


def play(env, model):
    observation = env.board
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False
    tot_reward = 0.0
    while not done:
        Q = model.predict(state)
        action = np.argmax(Q)
        observation, reward, done, legal= env.step(action)
        obs = np.expand_dims(observation, axis=0)
        state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
        tot_reward += reward
    print("Game ended! Total reward: {}".format(tot_reward))


if __name__ == "__main__":
    episode = 0
    env = Game(4)

    D = []
    model = build_model()
    for i in range(20):
        for j in range(20):
            observe(env, model, D)
            train(env, model, D, episode)
            episode += 1
            print("\n")
        D = []
        epsilon *= 0.9
        print("\n")
        play(env, model)
        model_json = model.to_json()
        with open(("model-data/model_%d.json" % episode), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(("model-data/model_%d.h5" % episode))
        print("Saved model at episode %d" % episode)
