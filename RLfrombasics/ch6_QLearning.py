import random
import numpy as np

class GridWrold():
    def __init__(self):
        self.x = 0
        self.y = 0

    def step(self, action):
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_up()
        elif action == 2:
            self.move_right()
        else:
            self.move_down()
        
        reward = -1
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_left(self):
        if self.y == 0:
            pass
        elif self.y == 3 and self.x in [0, 1, 2]:
            pass
        elif self.y == 5 and self.x in [2, 3, 4]:
            pass
        else:
            self.y -= 1

    def move_right(self):
        if self.y == 6:
            pass
        elif self.y == 1 and self.x in [0, 1, 2]:
            pass
        elif self.y == 3 and self.x in [2, 3, 4]:
            pass
        else:
            self.y += 1

    def move_up(self):
        if self.x == 0:
            pass
        elif self.x == 3 and self.y == 2:
            pass
        else:
            self.x -= 1

    def move_down(self):
        if self.x == 4:
            pass
        elif self.x == 1 and self.y == 4:
            pass
        else:
            self.x += 1

    def get_state(self):
        return (self.x, self.y)

    def is_done(self):
        if self.x == 4 and self.y == 6:
            return True
        return False

    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))
        self.eps = 0.9
        self.alpha = 0.1

    def select_action(self, s):
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,3)
        else:
            action_val = self.q_table[x, y, :]
            action = np.argmax(action_val)
        return action
    
    def update_table(self, transition):
        s, a, r, s_prime = transition
        x, y = s
        next_x, next_y = s_prime
        self.q_table[x, y, a] = self.q_table[x, y, a] + \
                                self.alpha * (r + np.amax(self.q_table[next_x, next_y, :]) - self.q_table[x, y, a])

    def anneal_eps(self):
        self.eps -= 0.01
        self.eps = max(self.eps, 0.2)

    def show_table(self):
        q_list = self.q_table.tolist()
        data = np.zeros((5, 7))
        for row_idx in range(len(q_list)):
            row = q_list[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)

def main():
    env = GridWrold()
    agent = QAgent()

    for n_epi in range(1000):
        done = False
        history = []

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s, a, r, s_prime))
            s = s_prime
        agent.anneal_eps()
    
    agent.show_table()


if __name__ == "__main__":
    main()