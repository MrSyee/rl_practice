"""DQN with tensorflow"""

import tensorflow as tf

layer = tf.contrib.layers


class DQN:
    def __init__(self, state_size, action_size, net_name):
        self.state_size = state_size
        self.action_size = action_size
        self.net_name = net_name

        with tf.variable_scope(self.net_name + 'input_layer'):
            self.input_ = tf.placeholder(tf.float32, shape=(None, self.state_size))
        self.q_value = self._build_network()

    def _build_network(self):
        with tf.variable_scope(self.net_name):
            fc1 = layer.fully_connected(
                inputs=self.input_,
                num_outputs=128,
                activation_fn=tf.nn.relu,
            )
            fc2 = layer.fully_connected(
                inputs=fc1,
                num_outputs=128,
                activation_fn=tf.nn.relu,
            )
            q_value = layer.fully_connected(
                inputs=fc2,
                num_outputs=self.action_size,
                activation_fn=None,
            )
            return q_value
