from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
import numpy as np

class Model:
    def __init__(self, action_space):
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=(210, 160, 3),filters=32, kernel_size=(8, 8), strides=4, activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"))
        self.model.add(Flatten())
        self.model.add(Dense(units=256, activation="relu"))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(units=action_space, activation="linear"))
        self.model.compile(optimizer="adam", loss="mse")

    def get_action(self, observation, verbose=False):
        prediction = self.model.predict(observation)
        if verbose:
            print(prediction)
        return np.argmax(prediction)

