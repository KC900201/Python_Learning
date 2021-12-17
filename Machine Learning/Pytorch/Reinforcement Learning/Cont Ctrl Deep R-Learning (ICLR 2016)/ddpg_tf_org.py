# Youtube reference for paper implementation - https://www.youtube.com/watch?v=GJJc1t0rtSU&ab_channel=freeCodeCamp.org

# Need a replay buffer class
# Need a class for a  target Q network
# Use batch-normalization (normalizing inputs)
# policy is deterministic, how to handle explorer exploit dilemma?
# deterministic policy ~ outputs the actual action instead of a probability
# will need a way to bound the actions to the env limits
# We have two actor and two critic networks, a target for each.
# Updates are soft, according to theta_prime = tau*theta + (1-tau)*theta_prime, with tau << 1
# The target actor is just the evaluation actor plus some noise process
# they used Ornstein Uhlenbeck (look up later!) -> need a class for the noise

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import random_uniform


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


class Actor():
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims, action_bound, batch_size=64,
                 chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = sess
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.chkpt_dir = chkpt_dir
        self.build_network()
        self.params = tf.Module.trainable_variables(scope=self.name)  # using tensorflow v2 module
        self.saver = tf.train.Checkpoint()  # migrate to tensorflow v2 (Saver is v1)
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')
        self.action_gradients = list(map(lambda x: tf.divide(x, self.batch_size), self.unnormalized_actor_gradients))
        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
        self.optimize = tf.keras.optimizers.Adam(self.lr). \
            apply_gradients(zip(self.action_gradients, self.params))
        # tf.compat.v1.trainable_variables

    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            self.action_gradient = tf.keras.Input(dtype=tf.float32,
                                                  shape=[None,
                                                         self.n_actions], )  # replace placeholder with keras.Input

            f1 = 1 / np.sqrt(self.fc1_dims)
            dense1 = tf.keras.layers.Dense(
                self.input, units=self.fc1_dims,
                kernel_initializer=random_uniform(shape=(-f1, f1)),  # tensorflow v2 model
                bias_initializer=random_uniform(shape=(-f1, f1))
            )
            batch1 = tf.keras.layers.BatchNormalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.keras.layers.Dense(
                self.input, units=self.fc2_dims,
                kernel_initializer=random_uniform(shape=(-f2, f2)),  # tensorflow v2 model
                bias_initializer=random_uniform(shape=(-f2, f2))
            )
            batch2 = tf.keras.layers.BatchNormalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            f3 = 0.003
            mu = tf.keras.layers.Dense(layer2_activation, units=self.n_actions,
                                       activation='tanh',
                                       kernel_initializer=random_uniform(shape=(-f3, f3)),
                                       bias_initializer=random_uniform(shape=(-f3, f3)))
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs, self.action_gradient: gradients})

    def save_checkpoint(self):
        print('... save checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)


class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = sess
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir
        self.build_network()
        self.params = tf.Module.trainable_variables(scope=self.name)  # using tensorflow v2 module
        self.saver = tf.train.Checkpoint()  # migrate to tensorflow v2 (Saver is v1)
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')
        self.optimize = tf.keras.optimizers.Adam(self.lr).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.input = tf.keras.Input(dtype=tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')  # replace placeholder with keras.Input
            self.actions = tf.keras.Input(dtype=tf.float32,
                                          shape=[None, self.n_actions],
                                          name='actions')
            self.q_target = tf.keras.Input(dtype=tf.float32,
                                           shape=[None, 1],
                                           name='targets')

            f1 = 1 / np.sqrt(self.fc1_dims)
            dense1 = tf.keras.layers.Dense(
                self.input, units=self.fc1_dims,
                kernel_initializer=random_uniform(shape=(-f1, f1)),  # tensorflow v2 model
                bias_initializer=random_uniform(shape=(-f1, f1))
            )
            batch1 = tf.keras.layers.BatchNormalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.keras.layers.Dense(
                self.input, units=self.fc2_dims,
                kernel_initializer=random_uniform(shape=(-f2, f2)),  # tensorflow v2 model
                bias_initializer=random_uniform(shape=(-f2, f2))
            )
            batch2 = tf.keras.layers.BatchNormalization(dense2)
            action_in = tf.layers.dense(self.actions, units=self.fc2_dims,
                                        activation='relu')

            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)

            f3 = 0.003
            self.q = tf.layers.dense(state_actions, units=1,
                                     kernel_initializers=random_uniform(-f3, f3),
                                     bias_initializer=random_uniform(-f3, f3),
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01))
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q, feed_dict={self.input: inputs,
                                                self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize, feed_dict={self.input: inputs,
                                                       self.actions: actions,
                                                       self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})

    def save_checkpoint(self):
        print('... save checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess,
                           layer1_size, layer2_size, env.action_space.high)
        self.critic = Critic(beta, n_actions, 'Critic', input_dims, self.sess,
                             layer1_size, layer2_size)
        self.target_actor = Actor(alpha, n_actions, 'TargetActor', input_dims,
                                  self.sess, layer1_size, layer2_size, env.action_space.high)
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims,
                                    self.sess, layer1_size, layer2_size)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_critic = [self.target_critic.params[i].assign(tf.multiply(self.critic.params[i], self.tau) \
                                                                  + tf.multiply(self.target_critic.params[i],
                                                                                1. - self.tau)) for i in
                              range(len(self.target_critic.params))]

        self.update_actor = [self.target_actor.params[i].assign(tf.multiply(self.actor.params[i], self.tau) \
                                                                + tf.multiply(self.target_actor.params[i],
                                                                              1. - self.tau)) for i in
                             range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())
        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.less.sun(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        noise = self.noise()
        mu_prime = mu + noise
        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, rewards, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        critic_value_ = self.target_critic.predict(new_state, self.target_actor.predict(new_state))

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = np.reshape(target, (self.batch_size, 1))
        _ = self.critic.train(state, action, target)
        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)
        self.actor.train(state, grads[0])

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
