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
        self.state_space_size = state_space_size
        self.buffer_cap = buffer_cap

        self.states0 = CircularBuffer(buffer_cap, np.float32, state_space_size)
        self.actions0 = CircularBuffer(buffer_cap, np.int64)
        self.states1 = CircularBuffer(buffer_cap, np.float32, state_space_size)
        self.actions1 = CircularBuffer(buffer_cap, np.int64)
        self.rewards = CircularBuffer(buffer_cap, np.float32)
        self.shaped_rewards0 = CircularBuffer(buffer_cap, np.float32)
        self.shaped_rewards1 = CircularBuffer(buffer_cap, np.float32)
        self.state_primes0 = CircularBuffer(buffer_cap, np.float32, state_space_size)
        self.state_primes1 = CircularBuffer(buffer_cap, np.float32, state_space_size)
        self.critic_values_prime = CircularBuffer(buffer_cap, np.float32)
        self.action_probs0 = CircularBuffer(buffer_cap, np.float32)
        self.action_probs1 = CircularBuffer(buffer_cap, np.float32)
        self.terminals = CircularBuffer(buffer_cap, np.int64)
        self.batch_size = batch_size
        self.advantages = None
        self.critic_losses = None

    def add(self, state0, action0, state1, action1, reward, sr0, sr1, state_prime0, state_prime1, critic_value_prime, action_probs0, action_probs1, terminate_episode):
        self.states0.add(state0)
        self.actions0.add(action0)
        self.states1.add(state1)
        self.actions1.add(action1)
        self.rewards.add(reward)
        self.shaped_rewards0.add(sr0)
        self.shaped_rewards1.add(sr1)
        self.state_primes0.add(state_prime0)
        self.state_primes1.add(state_prime1)
        self.critic_values_prime.add(critic_value_prime)
        self.action_probs0.add(action_probs0)
        self.action_probs0.add(action_probs1)
        self.terminals.add(terminate_episode)

    def sample(self):
        if self.advantages is None or self.critic_losses is None:
            raise Exception("Advantages and critic losses have not been set, error in training loop")

        current_buffer_size = self.states0.size()

        sample_count = min(self.batch_size, current_buffer_size)

        indices = np.random.choice(current_buffer_size, size=sample_count, replace=False)

        states0 = torch.tensor(self.states0.buffer[indices], dtype=torch.float32, requires_grad=True)
        actions0 = torch.tensor(self.actions0.buffer[indices], dtype=torch.int64)
        states1 = torch.tensor(self.states1.buffer[indices], dtype=torch.float32, requires_grad=True)
        actions1 = torch.tensor(self.actions1.buffer[indices], dtype=torch.int64)
        rewards = torch.tensor(self.rewards.buffer[indices], dtype=torch.float32, requires_grad=True)
        shaped_rewards0 = torch.tensor(self.shaped_rewards0.buffer[indices], dtype=torch.float32, requires_grad=True)
        shaped_rewards1 = torch.tensor(self.shaped_rewards1.buffer[indices], dtype=torch.float32, requires_grad=True)
        state_primes0 = torch.tensor(self.state_primes0.buffer[indices], dtype=torch.float32, requires_grad=True)
        state_primes1 = torch.tensor(self.state_primes1.buffer[indices], dtype=torch.float32, requires_grad=True)
        critic_values_prime = torch.tensor(self.critic_values_prime.buffer[indices], dtype=torch.float32, requires_grad=True)
        action_probs0 = torch.tensor(self.action_probs0.buffer[indices], dtype=torch.float32)
        action_probs1 = torch.tensor(self.action_probs1.buffer[indices], dtype=torch.float32)
        advantages = torch.tensor(self.advantages.buffer[indices], dtype=torch.float32, requires_grad=True)
        critic_losses = torch.tensor(self.critic_losses.buffer[indices], dtype=torch.float32, requires_grad=True)
        terminations = torch.tensor(self.terminals.buffer[indices], dtype=torch.int64)

        return (states0, actions0, states1, actions1, rewards, shaped_rewards0, shaped_rewards1, state_primes0,
                state_primes1, critic_values_prime, action_probs0, action_probs1, advantages, critic_losses,
                terminations)

    def get_all_states(self):
        return torch.tensor(self.states0.buffer, dtype=torch.float32), torch.tensor(self.states1.buffer, dtype=torch.float32, requires_grad=True)

    def get_entire_buffer_for_advantages_critic_losses(self):

        states0 = torch.tensor(self.states0.buffer, dtype=torch.float32, requires_grad=True)
        actions0 = torch.tensor(self.actions0.buffer, dtype=torch.int64)
        states1 = torch.tensor(self.states1.buffer, dtype=torch.float32, requires_grad=True)
        actions1 = torch.tensor(self.actions1.buffer, dtype=torch.int64)
        rewards = torch.tensor(self.rewards.buffer, dtype=torch.float32, requires_grad=True)
        shaped_rewards0 = torch.tensor(self.shaped_rewards0.buffer, dtype=torch.float32, requires_grad=True)
        shaped_rewards1 = torch.tensor(self.shaped_rewards1.buffer, dtype=torch.float32, requires_grad=True)
        state_primes0 = torch.tensor(self.state_primes0.buffer, dtype=torch.float32, requires_grad=True)
        state_primes1 = torch.tensor(self.state_primes1.buffer, dtype=torch.float32, requires_grad=True)
        critic_values_prime = torch.tensor(self.critic_values_prime.buffer, dtype=torch.float32, requires_grad=True)
        action_probs0 = torch.tensor(self.action_probs0.buffer, dtype=torch.float32)
        action_probs1 = torch.tensor(self.action_probs1.buffer, dtype=torch.float32)
        terminations = torch.tensor(self.terminals.buffer, dtype=torch.int64)

        return (states0, actions0, states1, actions1, rewards, shaped_rewards0, shaped_rewards1, state_primes0,
                state_primes1, critic_values_prime, action_probs0, action_probs1, terminations)

    def set_advantages_and_critic_loss(self, advantages, critic_losses):
        self.advantages = CircularBuffer(self.buffer_cap, np.float32)
        self.critic_losses = CircularBuffer(self.buffer_cap, np.float32)

        for i in range(len(advantages)):
            self.advantages.add(advantages[i])
            self.critic_losses.add(critic_losses[i])
