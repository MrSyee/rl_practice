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
        self.n_epsilon_decay = 200000
        self.epsilon = self.s_epsilon

        # place holder
        self.actions = tf.placeholder(tf.int32, shape=None)
        self.targets = tf.placeholder(tf.float32, shape=None)

        # network
        self.policy_net = DQN(self.state_size, self.action_size, net_name="policy_net")
        self.target_net = DQN(self.state_size, self.action_size, net_name="target_net")
        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

        # replay buffer
        self.buffer = ReplayBuffer(50000)

        # optimizer
        self.loss_op, self.train_op = self._build_op()

    def _build_op(self):
        """신경망 학습을 위한 Loss function과 Optimaizer를 정의합니다."""
        action_one_hot = tf.one_hot(self.actions, self.action_size, 1.0, 0.0)
        predict_q = tf.reduce_sum(tf.multiply(self.policy_net.q_value, action_one_hot), axis=1)

        loss_op = tf.reduce_mean(tf.square(self.targets - predict_q))
        train_op = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=0.95,
            momentum=0.95,
            epsilon=0.01
        ).minimize(loss_op)
        return loss_op, train_op

    def select_action(self, state):
        """epsilon-greedy로 action을 선택합니다."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = np.random.randint(self.action_size)
        else:
            state = np.expand_dims(state, axis=0)
            selected_action = self.sess.run(
                self.policy_net.q_value,
                feed_dict={self.policy_net.input_: state}
            )
            selected_action = np.argmax(selected_action, axis=1)
            selected_action = selected_action[0]

        # 매 step마다 epsilon을 줄여나갑니다.
        if self.epsilon >= self.e_epsilon:
            self.epsilon -= (self.s_epsilon - self.e_epsilon) / self.n_epsilon_decay
        return selected_action

    def update_model(self):
        """학습 네트워크를 학습합니다."""
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        target_q = self.target_net.q_value.eval({self.target_net.input_: next_states}, self.sess)
        target_q = np.max(target_q, axis=1)
        targets = rewards + self.discount_factor * target_q * (1. - dones)

        loss, _ = self.sess.run(
            [self.loss_op, self.train_op],
            feed_dict={self.policy_net.input_: states,
                       self.actions: actions,
                       self.targets: targets})

        return loss

    def update_target_network(self):
        """학습 네트웍의 변수의 값들을 타겟 네트웍으로 복사해서 타겟 네트웍의 값들을 최신으로 업데이트합니다."""
        copy_op = []

        main_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_net")
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_net")

        for main_q_var, target_var in zip(main_q_vars, target_vars):
            copy_op.append(target_var.assign(main_q_var.value()))
        self.sess.run(copy_op)

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

