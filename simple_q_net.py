import random
import sys

import gym
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential


# builds model with several dense layers
# adding option of convolutional layers for input to accomodate atari games
# noinspection PyShadowingNames
def build_model(with_conv=False):
    model = Sequential()
    model.add(Dense(20, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, init='uniform', activation='relu'))
    model.add(Dense(64, init='uniform', activation='relu'))
    model.add(Dense(16, init='uniform', activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(env.action_space.n, init='uniform', activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


# use built model to learn in set environment
# noinspection PyPep8Naming,PyPep8Naming,PyShadowingNames,PyShadowingNames
def observe_and_learn(model):
    # starting observation
    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False

    # observe for a set amount of timesteps and add the observations to memory
    # uses epsilon-greedy with epsilon annealed over time
    for t in range(observetime):
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            Q = model.predict(state)
            action = np.argmax(Q)
        observation_new, reward, done, info = env.step(action)
        obs_new = np.expand_dims(observation_new, axis=0)
        state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)
        D.append((state, action, reward, state_new, done))
        state = state_new
        if done:
            env.reset()
            obs = np.expand_dims(observation, axis=0)
            state = np.stack((obs, obs), axis=1)
    # finish observation

    # train the model with a preset minibatch size
    # model is trained with SARSA (state action reward state action) algorithm, with adam optimizer
    minibatch = random.sample(D, mb_size)

    inputs_shape = (mb_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, env.action_space.n))

    for i in range(mb_size):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]

        inputs[i:i + 1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)

        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

        loss = model.train_on_batch(inputs, targets)
        sys.stdout.write("\rEpisode = %s,Loss = %.5f" % (str(episode), loss))


# play/evaluate the model
# unlike observation and training, there is no probability of taking a random action while evaluating the model
# noinspection PyPep8Naming,PyShadowingNames
def play(model):
    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False
    tot_reward = 0.0
    while not done:
        env.render()  # Uncomment to see game running
        Q = model.predict(state)
        action = np.argmax(Q)
        observation, reward, done, info = env.step(action)
        obs = np.expand_dims(observation, axis=0)
        state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
        tot_reward += reward
    print('Game ended! Total reward: {}'.format(tot_reward))


if __name__ == '__main__':
    episode = 0
    env = gym.make('CartPole-v0')

    D = []
    observetime = 500
    epsilon = 0.95
    gamma = 0.9
    mb_size = 50
    model = build_model()
    mode = input("Input mode: ")
    if mode.upper() == 'TRAIN':
        for i in range(10):
            for j in range(20):
                observe_and_learn(model)
                episode += 1
                print('\n')
                D = []
            epsilon *= 0.9
            print('\n')
            play(model)
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        print("Saved Model")
    elif mode.upper() == "PLAY":
        model = build_model()
        model.load_weights('model.h5')
        for i in range(20):
            play(model)
