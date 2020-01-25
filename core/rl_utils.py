from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size=100000, random_seed=123):
        self._buffer_size = buffer_size
        self._count = 0
        self._buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, s2, r, t):
        experience = (s, a, s2, r, t)
        if self._count < self._buffer_size:
            self._buffer.append(experience)
            self._count += 1
        else:
            self._buffer.popleft()
            self._buffer.append(experience)

    def sample_batch(self, batch_size):
        batch = []
        if self._count < batch_size:
            batch = random.sample(self._buffer, self._count)
        else:
            batch = random.sample(self._buffer, batch_size)
        return batch

    def clear(self):
        self._buffer.clear()
        self._count = 0
