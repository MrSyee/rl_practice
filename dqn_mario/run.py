"""Run DQN agent"""

import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import argparse
import tensorflow as tf

from agent import DQNAgent
from wrapper import wrap_mario

parser = argparse.ArgumentParser(description="Run dqn agent")
parser.add_argument(
    "--test", dest="test", action="store_true", help="test mode (no training)"
)
parser.add_argument(
    "--render", dest="render", action="store_true", help="test mode (no training)"
)
args = parser.parse_args()

EPISODE = 500
TARGET_UPDATE = 100

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env = wrap_mario(env)
"""
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]"""

state_size = env.observation_space.shape
action_size = env.action_space.n
print("state_size", state_size)
print("action_size", action_size)

# create dqn agent
sess = tf.Session()
dqn = DQNAgent(sess, state_size, action_size)
sess.run(tf.global_variables_initializer())


def train():
    for n_episode in range(1, EPISODE + 1):
        state = env.reset()
        total_step = 1
        total_reward = []
        total_loss = []
        done = False

        while not done:
            action = dqn.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if args.render:
                env.render()

            if total_step % train_period == 0:
                loss = dqn.update_model()
                total_loss.append(loss)
            total_reward.append(reward)

    env.close()


def test():
    pass


if __name__ == "__main__":
    if args.test:
        test()
    else:
        train()
