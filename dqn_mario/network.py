"""DQN with tensorflow"""

import tensorflow as tf

layer = tf.contrib.layers


class DQN:
    def __init__(self, state_size, action_size, net_name):
        self.state_size = state_size
        self.action_size = action_size
        self.net_name = net_name

        with tf.variable_scope(self.net_name + 'input_layer'):
            self.input_ = tf.placeholder(tf.float32, shape=(
                None, self.state_size[0], self.state_size[1], self.state_size[2])
            )
        self.q_value = self._build_network()

    def _build_network(self):
        with tf.variable_scope(self.net_name):
            conv1 = layer.conv2d(
                inputs=self.input_,
                num_outputs=32,
                kernel_size=[8, 8],
                stride=[4, 4],
            )
            conv2 = layer.conv2d(
                inputs=conv1,
                num_outputs=64,
                kernel_size=[4, 4],
                stride=[2, 2],
            )
            conv3 = layer.conv2d(
                inputs=conv2,
                num_outputs=64,
                kernel_size=[3, 3],
            )
            conv3_flatten = layer.flatten(conv3)

            fc1 = layer.fully_connected(
                inputs=conv3_flatten,
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
