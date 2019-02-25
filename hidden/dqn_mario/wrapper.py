import numpy as np
from collections import deque
import gym
import cv2

IMG_SIZE = 84


def _process_frame_mario(frame):
    if frame is not None:  # for future meta implementation
        img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        x_t = np.expand_dims(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), axis=-1)
        x_t.astype(np.uint8)
    else:
        x_t = np.zeros((IMG_SIZE, IMG_SIZE, 1))

    return x_t


class ProcessFrameMario(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrameMario, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, IMG_SIZE, IMG_SIZE))
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
    def __init__(self, env=None, skip=4, shape=(IMG_SIZE, IMG_SIZE)):
        super(BufferSkipFrames, self).__init__(env)
        self.counter = 0
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(IMG_SIZE, IMG_SIZE, 4))
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
        frame = np.reshape(frame, (IMG_SIZE, IMG_SIZE, 4))
        return frame, total_reward, done, info

    def _reset(self):
        self.buffer.clear()
        obs = self.env.reset()
        for i in range(self.skip):
            self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (IMG_SIZE, IMG_SIZE, 4))
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


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info['life']
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            self.done = True
        self.lives = lives
        return obs, reward, self.done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs


def wrap_mario(env):
    # assert 'SuperMarioBros' in env.spec.id
    env = ProcessFrameMario(env)
    env = NormalizedEnv(env)
    env = BufferSkipFrames(env)
    env = EpisodicLifeEnv(env)

    return env
