import random
from collections import deque

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

class Agent():
    def __init__(self, state_dim, is_eval=False):

        self.model_type = 'DQN'
        self.state_dim = state_dim #input size = window size + portfolio state(3)
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 30

        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995 # decrease exploration rate as the agent becomes good at trading
        self.is_eval = is_eval #Train or test
        self.model = load_model('DQN_ep10.h5') if is_eval else self.model()

        # self.tensorboard = TensorBoard(log_dir='./logs/DQN_tensorboard', update_freq=90)
        # self.tensorboard.set_model(self.model)


    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model

    def reset(self):
        self.epsilon = 1.0 # reset exploration rate

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    #exploration function
    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim) #choose a random action
        options = self.model.predict(state) #predict
        return np.argmax(options[0])

    def experience_replay(self):
        # retrieve recent buffer_size long memory
        #last <buffer_size> number of states into mini batch
        mini_batch = [self.memory[i] for i in range(len(self.memory) - self.buffer_size + 1, len(self.memory))]

        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                Q_target_value = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                Q_target_value = reward
            next_actions = self.model.predict(state)
            next_actions[0][np.argmax(actions)] = Q_target_value
            history = self.model.fit(state, next_actions, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]
