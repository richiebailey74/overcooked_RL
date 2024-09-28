import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import gym
from .Network import DeepQNetwork
from .ExperienceBuffer import ExperienceBuffer
import gc
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


class DeepQLearningAgent:

    def __init__(
            self,
            env,
            buffer_cap,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
            learning_rate,
            batch_size=1,
            loss_function=torch.nn.MSELoss()
    ):
        self.Q = None  # NN for generating predicted Q's
        self.Q_targ = None  # NN for generating target Q's
        self.define_deep_nets(env.observation_space.shape[0], env.action_space.n)

        # use lazy delete double data structure to maintain experience buffer
        self.experience_buffer = ExperienceBuffer(buffer_cap, batch_size, env.observation_space.shape[0])

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        np.random.seed(931)

        self.reward_array = []
        self.reward_array_val = []

    def define_deep_nets(self, state_dimensionality, action_dimensionality):
        self.Q = DeepQNetwork(state_dimensionality, action_dimensionality)
        self.Q_targ = copy.deepcopy(self.Q)  # we want a NN with the same parameters to start to generate the Q targs

    # copy weights from target network into the prediction network
    def update_q_target_network(self):
        self.Q_targ.load_state_dict(copy.deepcopy(self.Q.state_dict()))

    def update_q_online_network(self):
        states, actions, rewards, state_primes, terminations = self.experience_buffer.sample()
        self.optimizer.zero_grad()
        q_values = self.Q(states).gather(1, actions.unsqueeze(-1)).squeeze(
            -1)  # select Q value for action taken, along with necessary dimension operations
        q_values_prime = self.Q_targ(state_primes).max(1)[0].detach() * (
                    1 - terminations)  # turn to 0 if it is a terminal state
        q_targets = rewards + (self.gamma * q_values_prime)
        loss = self.loss_function(q_values, q_targets)
        loss.backward()
        self.optimizer.step()
        del states
        del actions
        del rewards
        del state_primes
        del terminations

    # uses online Q network to generate output for the passed state and then the max value's index (correct action)
    def get_max_q_action_online(self, state):
        # use torch context manager to disable gradient computation to save computations
        with torch.no_grad():
            output = self.Q(torch.tensor(state, dtype=torch.float32).unsqueeze(0))

        _, max_q_ind = torch.max(output, dim=1)
        return max_q_ind

    # uses target Q network to generate the target Q value for the discounted reward/target
    def get_max_q_over_actions_target(self, state):
        # use torch context manager to disable gradient computation to save computations
        with torch.no_grad():
            output = self.Q_targ(torch.tensor(state, dtype=torch.float32).unsqueeze(0))

        max_q_value, _ = torch.max(output, dim=1)
        return max_q_value

    def select_action_online(self, state, env):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = self.get_max_q_action_online(state)
        return action

    # define an exploration function at compile time, abstract away the decaying
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        # self.epsilon = max(self.epsilon_min, self.epsilon - .00175)
