"""DQN with tensorflow"""

import tensorflow as tf

layer = tf.contrib.layers


class DQN:
    def __init__(self, state_size, action_size, net_name):
        self.state_size = state_size
        self.action_size = action_size
        self.net_name = net_name

        with tf.variable_scope(self.net_name + 'input_layer'):
            self.input_ =  # placeholder 구현
        self.q_value = self._build_network()

    def _build_network(self):
        with tf.variable_scope(self.net_name):
            """
            network 구현 
            """

            return q_value
