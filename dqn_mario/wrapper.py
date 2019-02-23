import numpy as np
from collections import deque
import gym
import cv2


def _process_frame_mario(frame):
    if frame is not None:  # for future meta implementation
        img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        x_t = np.expand_dims(cv2.resize(img, (84, 84)), axis=-1)
        x_t.astype(np.uint8)
    else:
        x_t = np.zeros((84, 84, 1))
    return x_t


class ProcessFrameMario(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrameMario, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84))
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40

    def _step(self, action):
        '''
            Implementing custom rewards
                Time = -0.1
                Distance = +1 or 0
                Player Status = +/- 5
                Score = 2.5 x [Increase in Score]
                Done = +50 [Game Completed] or -50 [Game Incomplete]
        '''
        obs, reward, done, info = self.env.step(action)

        return _process_frame_mario(obs), reward, done, info

    def _reset(self):
        return _process_frame_mario(self.env.reset())

    def change_level(self, level):
        self.env.change_level(level)


class BufferSkipFrames(gym.Wrapper):
    def __init__(self, env=None, skip=4, shape=(84, 84)):
        super(BufferSkipFrames, self).__init__(env)
        self.counter = 0
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 4))
        self.skip = skip
        self.buffer = deque(maxlen=self.skip)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        counter = 1
        total_reward = reward
        self.buffer.append(obs)

        for i in range(self.skip - 1):
            if not done:
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                counter += 1
                self.buffer.append(obs)
            else:
                self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (84, 84, 4))
        return frame, total_reward, done, info

    def _reset(self):
        self.buffer.clear()
        obs = self.env.reset()
        for i in range(self.skip):
            self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (84, 84, 4))
        return frame

    def change_level(self, level):
        self.env.change_level(level)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        if observation is not None:  # for future meta implementation
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                              observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                             observation.std() * (1 - self.alpha)

            unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
            unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

            return (observation - unbiased_mean) / (unbiased_std + 1e-8)

        else:
            return observation

    def change_level(self, level):
        self.env.change_level(level)


def wrap_mario(env):
    # assert 'SuperMarioBros' in env.spec.id
    env = ProcessFrameMario(env)
    env = NormalizedEnv(env)
    env = BufferSkipFrames(env)
    return env
