import sys
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability import distributions as tfd

class ContinuousA2C(tf.keras.Model):
    def __init__(self, action_size):
        super(ContinuousA2C, self).__init__()
        self.action_fc1 = Dense(24, activation='tanh')
        self.action_mu = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.action_sigma = Dense(action_size, activation='sigmoid', kernel_initializer=RandomUniform(-1e-3, 1e-3))

        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.action_fc1(x)
        mu = self.action_mu(actor_x)
        sigma = self.action_sigma(actor_x)
        sigma = sigma + 1e-5

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)

        return mu, sigma, value

class ContinuousA2CAgent:
    def __init__(self, action_size, max_action):
        self.render = True

        self.action_size = action_size
        self.max_action = max_action

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = ContinuousA2C(self.action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)

    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            mu, sigma, value = self.model(state)
            _, _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            advantage = tf.stop_gradient(target - value[0])
            dist = tfd.Normal(loc=mu, scale=sigma)
            action_prob = dist.prob([action])[0]
            cross_entropy = -tf.math.log(action_prob + 1e-5)
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            loss = 0.1 * actor_loss + critic_loss

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return loss, sigma


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    gym.envs.register(
        id='CartPoleContinuous-v0',
        entry_point='env:ContinuousCartPoleEnv',
        max_episode_steps=500,
        reward_threshold=475.0)

    env = gym.make('CartPoleContinuous-v0')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # 액터-크리틱(A2C) 에이전트 생성
    agent = ContinuousA2CAgent(action_size, max_action)
    scores, episodes = [], []
    score_avg = 0

    num_episode = 1000
    for e in range(num_episode):
        done = False
        score = 0
        loss_list, sigma_list = [], []
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -1 보상
            score += reward
            reward = 0.1 if not done or score == 500 else -1

            # 매 타임스텝마다 학습
            loss, sigma = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            sigma_list.append(sigma)
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f} | sigma: {:.3f}".format(
                      e, score_avg, np.mean(loss_list), np.mean(sigma)))

                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")

                # 이동 평균이 400 이상일 때 종료
                if score_avg > 400:
                    agent.model.save_weights("./save_model/model", save_format="tf")
                    sys.exit()