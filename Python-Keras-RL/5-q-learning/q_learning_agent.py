import numpy as np
import random
from environment import Env
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions):
        # 행동 = [0, 1, 2, 3] 순서대로 상, 하, 좌, 우
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state):
        state, next_state = str(state), str(next_state)
        q_1 = self.q_table[state][action]
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.step_size * (q_2 - q_1)

    # 큐함수에 의거하여 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state = str(state)
            q_list = self.q_table[state]
            action = self.arg_max(q_list)
        return action

    @staticmethod
    def arg_max(q_list):
        max_idx_list = np.argwhere(q_list == np.amax(q_list))
        max_idx_list = max_idx_list.flatten().tolist()
        return random.choice(max_idx_list)

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()

        while True:
            env.render()
            action = agent.get_action(str(state))

            next_state, reward, done = env.step(action)
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state

            env.print_value_all(agent.q_table)

            if done:
                break

