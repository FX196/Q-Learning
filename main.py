import random
import sys

import gym
import numpy as np
from keras.layers import Dense, Conv2D
from keras.models import Sequential

# hyperparameters
observe_time = 5000
episodes = 20
epsilon = 1.0
mb_size = 100
gamma = 0.9
D = []


# model for playing atari games, with conv layers for extracting positional information
# still in progress

# builds model with 3 conv layers followed by 2 dense layers, dropout optional
def build_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(84, 84, 4), \
                     filters=32, \
                     kernel_size=8, \
                     strides=4, \
                     padding="same", \
                     activation="relu"))
    model.add(Conv2D(filters=64, \
                     kernel_size=4, \
                     strides=2, \
                     padding="same", \
                     activation="relu"))
    model.add(Conv2D(filters=64, \
                     kernel_size=3, \
                     strides=1, \
                     padding="same", \
                     activation="relu"))
    model.add(Dense(units=512, \
                    activation='relu'))
    model.add(Dense(units=18, \
                    activation='relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model


# takes maximum value for every pixel over two frames to eliminate shimmering from sprites, this is a problem with
# the atari game console as it can only display a certain number of sprites per frame
# this function also converts RGB values to grayscale to decrease the amount of parameters that need to be learned
def process_frames(old, new):
    frame = np.maximum(old, new)
    frame = frame[25:185][:][:]
    frame_new = np.zeros(shape=(80, 80))
    for i in range(80):
        for j in range(80):
            frame_new[i][j] = np.average(frame[i * 2][j * 2])
    frame_new = np.pad(frame_new, 2, mode='edge')
    return frame_new


# trains network with same procedure as in simple_q_net.py, except that actions are repeated over 4 frames to decrease runtime
# this is possible due to the fact that the atari counsle displays at 60Hz, so little change can happen in 4 frames (67ms)
def train(model, env):
    frame_old = env.reset()
    processed_frame = process_frames(frame_old, frame_old)
    state = []
    for i in range(4):
        state.append(processed_frame)
    done = False
    for _ in range(episodes):
        for _ in range(observe_time):
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                Q = model.predict(state)
                action = np.argmax(Q)

            # repeat over 4 frames
            frame_new, reward, done, info = env.step(action)
            for _ in range(3):
                env.step(action)

            # update state to 4 most recent frames
            processed_frame = process_frames(frame_old, frame_new)
            state_new = state[1:4].append(processed_frame)

            reward /= abs(reward)+0.00001
            D.append((state, action, reward, state_new, done))
            state = state_new
            frame_old = frame_new
            if done:
                frame_old = env.reset()
                state = process_frames(frame_old, frame_old)
                done = False

        minibatch = random.sample(D, mb_size)
        inputs_shape = mb_size
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((mb_size, 18))

        for i in range(mb_size):
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            state_new = minibatch[i][3]
            done = minibatch[i][4]

            inputs[i:i + 1] = np.expand_dims(state, axis=0)
            targets[i] = model.predict(state)
            # noinspection PyPep8Naming
            Q_sa = model.predict(state_new)

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + gamma * np.max(Q_sa)

            loss = model.train_on_batch(inputs, targets)
            sys.stdout.write("\rLoss = %.5f" % loss)


def play(model, env):
    frame_old = env.reset()
    processed_frame = process_frames(frame_old, frame_old)
    state = []
    tot_reward = 0.0
    for i in range(4):
        state.append(processed_frame)
    done = False
    while not done:
        # uncomment next line to see game play
        # env.render()
        Q = model.predict(state)
        action = np.argmax(Q)

        # repeat over 4 frames
        frame_new, reward, done, info = env.step(action)
        for _ in range(3):
            env.step(action)

        # update state to 4 most recent frames
        processed_frame = process_frames(frame_old, frame_new)
        state_new = state[1:4].append(processed_frame)
        tot_reward += reward
    print('Game ended! Total reward: {}'.format(reward))


if __name__ == '__main__':
    episode = 0
    env = gym.make('Breakout-v0')
    model = build_model()
    train(model, env)