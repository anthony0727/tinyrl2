from collections import deque

import numpy as np


class StaticReplayBuffer:
    """faster, smaller
    efficient in shaping, indexing.
    can access attrs by its name
    """

    def __init__(self, num_envs, capacity):
        self.num_envs = num_envs
        self.capacity = capacity

        self.obs, self.action, self.reward, self.done = None, None, None, None

    def allocate(self, sample_transition):
        for field in sample_transition._fields:
            elem = getattr(sample_transition, field)
            shape = (self.capacity, self.num_envs, elem.shape)
            data = np.zeros(shape, dtype=elem.dtype)
            setattr(self, field, data)

    def push(self, obs, action, reward, done):
        self.obs[self.idx] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.done[self.idx] = done


class DynamicReplayBuffer:
    """slower, larger
    maximum data size up to the machine's memory.
    can't access attrs by its name.
    """

    def __init__(self, num_envs):
        self.num_envs = num_envs
        # data
        self._envs = None

    def allocate(self, sample_transition):
        self._envs = [deque()] * self.num_envs

    @property
    def data(self):
        # TODO: check if += duplicates memory
        ret = deque()
        for env_data in self._envs:
            ret += env_data

        return ret

    # field order matters
    def push(self, obs, action, reward, done):
        assert self._envs is not None, 'you must call .allocate() first'
        # push to i_th env
        for i, transition in enumerate(zip(obs, action, reward, done)):
            self._envs[i].append(transition)
