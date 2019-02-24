"""Run DQN agent"""

import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import argparse
import tensorflow as tf
import numpy as np

from agent import DQNAgent
from wrapper import wrap_mario

parser = argparse.ArgumentParser(description="Run dqn agent")
parser.add_argument(
    "--test", action="store_true", help="test mode (no training)"
)
parser.add_argument(
    "--render", action="store_true", help="test mode (no training)"
)
parser.add_argument(
    "--load-from", type=str, help="test mode (no training)"
)
parser.set_defaults(load_from=None)
args = parser.parse_args()

EPISODE = 500
TARGET_UPDATE = 100
TRAIN_START = 0
SAVE_PERIOD = 5
CHECKPOINT_NAME = "dqn"

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

if args.load_from is not None:
    dqn.load_model(args.load_from)

sess.run(tf.global_variables_initializer())


def train():
    for n_episode in range(1, EPISODE + 1):
        state = env.reset()
        total_step = 1
        total_reward = 0
        total_loss = []
        done = False

        while not done:
            action = dqn.select_action(state)
            next_state, reward, done, _ = env.step(action)
            dqn.buffer.add(state, action, reward, next_state, done)
            total_reward += reward

            if args.render:
                env.render()

            # train model
            if total_step > TRAIN_START and len(dqn.buffer) > dqn.batch_size:
                loss = dqn.update_model()
                total_loss.append(loss)

                # update target network
                if total_step % TARGET_UPDATE == 0:
                    dqn.update_target_network()

            state = next_state
            total_step += 1

        # save model
        if total_step > TRAIN_START and n_episode % SAVE_PERIOD == 0:
            dqn.save_model(CHECKPOINT_NAME + str(n_episode))

        mean_loss = np.mean(total_loss)
        print("[Episode {}] step {}, reward {}, loss {:.4f}, epsilon {:.4f}".format(
            n_episode, total_step, total_reward, mean_loss, dqn.epsilon
        ))

    env.close()


def test():
    dqn.epsilon = 0

    for n_episode in range(1, EPISODE + 1):
        state = env.reset()
        total_step = 1
        total_reward = 0
        done = False

        while not done:
            action = dqn.select_action(state)
            next_state, reward, done, info = env.step(action)
            print(info)
            total_reward += reward

            if args.render:
                env.render()

            state = next_state
            total_step += 1

        print("[Episode {}] step {}, reward {}".format(
            n_episode, total_step, total_reward
        ))

    env.close()


if __name__ == "__main__":
    if args.test:
        test()
    else:
        train()
