# 바닥부터 배우는 강화학습 (저자: 노승은)
# 몬테카를로 학습 구현 

"""
4가지 구현 요소
1. 환경 : 에이전트의 액션을 받아 상태변이를 일으키고, 보상을 줌
2. 에이전트 : 4방향 랜덤 정책을 이용해 움직임
3. 경험 쌓는 부분 : 에이전트가 환경과 작용하며 데이터를 축적
4. 학습하는 부분  : 쌓인 경험을 통해 테이블을 업데이트

스텝마다 보상은 -1로 고정
"""

import random
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.x = 0
        self.y = 0

    def step(self, action):
        # 북, 남, 동, 서
        if action == 0:
            self.move_up()
        if action == 1:
            self.move_down()
        if action == 2:
            self.move_right()
        if action == 3:
            self.move_left()
        
        reward = -1
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def move_right(self):
        self.y += 1
        if self.y > 3:
            self.y = 3

    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else:
            return False

    def get_state(self):
        return (self.x, self.y)

    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class Agent:
    def __init__(self):
        pass

    def move(self):
        policy = random.random()
        if policy < 0.25:
            action = 0
        elif policy < 0.5:
            action = 1
        elif policy < 0.75:
            action = 2
        else:
            action = 3        
        return action

def main():
    env = Environment()
    agent = Agent()

    data = list([0, 0, 0, 0] for _ in range(4))
    gamma = 1.0
    alpha = 0.0001

    for k in range(50000):
        done = False
        history = []
        while not done:
            action = agent.move()
            (x, y), reward, done = env.step(action)
            history.append((x,y, reward))
        env.reset()

        cum_reward = 0
        for trasition in history[::-1]:
            x, y, reward = trasition
            data[x][y] = data[x][y] + alpha * (cum_reward - data[x][y])
            print(data[x][y])
            cum_reward = reward + gamma * cum_reward
        
    for row in data:
        print(row)

if __name__ == "__main__":
    main()