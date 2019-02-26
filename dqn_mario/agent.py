"""DQN Agent"""

import tensorflow as tf
import numpy as np

from network import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, sess, state_size, action_size):
        self.sess = sess

        self.state_size = state_size
        self.action_size = action_size

        # hyper parameter
        self.batch_size = 32
        self.discount_factor = 0.99
        self.learning_rate = 0.00025

        # epsilon
        self.s_epsilon = 1.0
        self.e_epsilon = 0.01
        self.n_epsilon_decay = 100000
        self.epsilon = self.s_epsilon

        # replay buffer
        self.buffer = ReplayBuffer(50000)

        # place holder
        self.actions = tf.placeholder(tf.int32, shape=None)
        self.targets = tf.placeholder(tf.float32, shape=None)

        # network
        self.policy_net = DQN({})
        self.target_net = DQN({})
        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

        # optimizer
        self.loss_op, self.train_op = self._build_op()

    def _build_op(self):
        """신경망 학습을 위한 Loss function과 Optimaizer를 정의합니다."""

    def select_action(self, state):
        """epsilon-greedy로 action을 선택합니다."""

    def update_model(self):
        """학습 네트워크를 학습합니다."""

    def update_target_network(self):
        """학습 네트웍의 변수의 값들을 타겟 네트웍으로 복사해서 타겟 네트웍의 값들을 최신으로 업데이트합니다."""


    def save_model(self, filename):
        """Save model."""
        saver = tf.train.Saver()
        path = "./save/" + filename + ".ckpt"
        save_path = saver.save(self.sess, path)
        print("[Model saved in path: %s !!!]" % save_path)

    def load_model(self, filename):
        """Load model."""
        saver = tf.train.Saver()
        path = "./save/" + filename + ".ckpt"
        saver.restore(self.sess, path)
        print("[Model restored !!!]")

