import sys
import gym
import pylab
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform

class DQN(tf.keras.Model):
    def __init__(self, action_szie):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_szie,
                            kernel_initializer=RandomUniform(-1e-3, 1e-3))
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.model = DQN(action_size)
        self.model.load_weights("./save_model/model")

    def get_action(self, state):
        q_value = self.model(state)
        return np.argmax(q_value[0])


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    num_episode = 10
    for e in range(num_episode):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            env.render()

            action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            state = next_state

            if done:
                print("episode: {:3d} | score: {:.3f} ".format(e, score))
            