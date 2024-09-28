import numpy as np
import torch


class CircularBuffer:
    def __init__(self, buffer_cap, dt, second_dim=None):
        self.buffer_cap = buffer_cap
        if second_dim is None:
            self.buffer = np.zeros(buffer_cap, dtype=dt)
        else:
            self.buffer = np.zeros((buffer_cap, second_dim), dtype=dt)
        self.first = None
        self.last = None
        self.hit_capacity = False

    def add(self, sample):
        if self.first is None:
            self.buffer[0] = sample
            self.first = 0
            self.last = 0
        else:
            self.last += 1
            self.last %= self.buffer_cap
            self.buffer[self.last] = sample
            if self.first == self.last: self.first += 1
            if self.last == self.buffer_cap - 1: self.hit_capacity = True

    def size(self):
        if self.first is None:
            return 0
        elif self.hit_capacity:
            return self.buffer_cap
        elif self.last >= self.first:
            return self.last - self.first + 1
        else:
            return self.buffer_cap - self.first + self.last + 1


class ExperienceBuffer:
    def __init__(self, buffer_cap, batch_size, state_space_size):
        self.states = CircularBuffer(buffer_cap, np.float32, state_space_size)
        self.actions = CircularBuffer(buffer_cap, np.int64)
        self.rewards = CircularBuffer(buffer_cap, np.float32)
        self.state_primes = CircularBuffer(buffer_cap, np.float32, state_space_size)
        self.terminals = CircularBuffer(buffer_cap, np.int64)
        self.batch_size = batch_size

    def add(self, state, action, reward, state_prime, terminate_episode):
        self.states.add(state)
        self.actions.add(action)
        self.rewards.add(reward)
        self.state_primes.add(state_prime)
        self.terminals.add(terminate_episode)

    def sample(self):
        current_buffer_size = self.states.size()

        sample_count = min(self.batch_size, current_buffer_size)

        indices = np.random.choice(current_buffer_size, size=sample_count, replace=False)

        states = torch.tensor(self.states.buffer[indices], dtype=torch.float32)
        actions = torch.tensor(self.actions.buffer[indices], dtype=torch.int64)
        rewards = torch.tensor(self.rewards.buffer[indices], dtype=torch.float32)
        state_primes = torch.tensor(self.state_primes.buffer[indices], dtype=torch.float32)
        terminations = torch.tensor(self.terminals.buffer[indices], dtype=torch.int64)

        return states, actions, rewards, state_primes, terminations
