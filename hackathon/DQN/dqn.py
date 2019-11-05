import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model
import numpy as np
import random
from collections import deque
import gym

env = gym.make('CartPole-v0')
num_action = env.action_space.n
state_size = env.observation_space.shape[0]

class DQNModel(Model):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.layer1 = Dense(64, activation='relu')
        self.layer2 = Dense(64, activation='relu')
        self.value = Dense(num_action)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)
        value = self.value(layer2)
        return value


class Trainer:
    def __init__(self):
        # hyper parameters
        self.lr = 0.001
        self.lr2 = 0.001
        self.gamma = 0.99

        # create model and target model
        self.dqn_model = DQNModel()
        self.dqn_target = DQNModel()
        self.opt = optimizers.Adam(lr=self.lr)

        # epsilon-greedy action selection
        # with decaying epsilon
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.state_size = state_size

        self.memory = deque(maxlen=2000)

    def update_target(self):
        """
        Updates the target network
        :return: none
        """
        self.dqn_target.set_weights(self.dqn_model.get_weights())

    def get_action(self, state):
        """
        Select an action given a state. Implements epsilon-greedy strategy
        :param state: state input
        :return: selected action
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(num_action)
        else:
            q_value = self.dqn_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            return np.argmax(q_value[0])

    def append_tuple(self, state, action, reward, next_state, done):
        """
        Append a tuple to experience buffer
        :param state: obsvered state
        :param action: executed action
        :param reward: reward recieved
        :param next_state: state reached after exectuting a in s
        :param done: whether or not environment reached a terminal state
        :return: none
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """
        Trains the DQN model.
        :return: none
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # experience replay
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        # split batch
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        dqn_variable = self.dqn_model.trainable_variables

        # compute gradients and uptadate model parameters
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            # get targets
            target = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            target_val = self.dqn_target(tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32))

            target = np.array(target)
            target_val = np.array(target_val)

            for i in range(self.batch_size):
                next_v = np.array(target_val[i]).max()
                # q-learning
                if dones[i]:
                    # if terminal state just take the reward
                    target[i][actions[i]] = rewards[i]
                else:
                    # if not terminal state we also take discounted future reward into account
                    target[i][actions[i]] = rewards[i] + self.gamma * next_v

            values = self.dqn_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
            error = tf.square(values - target) * 0.5
            error = tf.reduce_mean(error)

        dqn_grads = tape.gradient(error, dqn_variable)
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))

    def run(self):

        t_end = 500
        epi = 100000

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for e in range(epi):
            total_reward = 0
            for t in range(t_end):
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                #env.render()

                if t == t_end :
                    done = True
                if t < t_end and done :
                    reward = -1

                total_reward += reward
                self.append_tuple(state, action, reward, next_state, done)

                if len(self.memory) >= self.train_start:
                    self.train()

                total_reward += reward
                state = next_state

                if done:
                    self.update_target()
                    print("e : ", e, " reward : ", total_reward, " step : ", t)
                    env.reset()
                    break


if __name__ == '__main__':
    DQN = Trainer()
    DQN.run()

