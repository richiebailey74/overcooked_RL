from .Networks import DeepActorNetwork, DeepCriticNetwork
import torch
import torch.nn.functional as F
from .ExperienceBuffer import ExperienceBuffer
import copy
import numpy as np


class CTDEAgent:
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
        self.actor = None
        self.actor_targ = None
        self.critic = None
        self.critic_targ = None
        self.define_deep_nets(env.observation_space.shape[0], env.action_space.n)
        self.ohe_num_classes = env.action_space.n

        self.experience_buffer = ExperienceBuffer(buffer_cap, batch_size, env.observation_space.shape[0])

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.loss_function_critic = loss_function
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

    def define_deep_nets(self, state_dimensionality, action_dimensionality):
        self.actor = DeepActorNetwork(state_dimensionality, action_dimensionality)
        self.actor_targ = copy.deepcopy(self.actor)

        self.critic = DeepCriticNetwork((state_dimensionality * 2) + (action_dimensionality * 2), 1)
        self.critic_targ = copy.deepcopy(self.critic)

    def update_target_networks(self):
        self.actor_targ.load_state_dict(copy.deepcopy(self.actor.state_dict()))
        self.critic_targ.load_state_dict(copy.deepcopy(self.critic.state_dict()))

    def update_actor_critic_networks(self):
        # sample experience buffer
        states0, actions0, states1, actions1, rewards, shaped_rewards0, shaped_rewards1, state0_primes, state1_primes, terminations = self.experience_buffer.sample()

        # get critic inputs
        critic_input, critic_target_input = self.get_critic_inputs(states0, actions0, states1, actions1, state0_primes, state1_primes)

        # get Q values from critic and critic target
        critic_q_values = self.critic(critic_input)
        critic_prime_q_values = self.critic_targ(critic_target_input).detach()

        # construct the q targets (unsqueeze rewards to make dimensions compatible)
        critic_q_targets = (rewards + shaped_rewards0 + shaped_rewards1).unsqueeze(-1) + (self.gamma * critic_prime_q_values)

        # update critic
        self.update_critic(critic_q_values, critic_q_targets)

        critic_q_values = self.critic(critic_input)
        critic_prime_q_values = self.critic_targ(critic_target_input).detach()
        critic_q_targets = (rewards + shaped_rewards0 + shaped_rewards1).unsqueeze(-1) + (self.gamma * critic_prime_q_values)
        # update actor
        self.update_actor(critic_q_values.detach(), critic_q_targets, states0, actions0, states1, actions1)

    def get_critic_inputs(self, states0, actions0, states1, actions1, state0_primes, state1_primes):
        # one hot encode the actions for input to the critic network
        actions0_ohe = F.one_hot(actions0, num_classes=self.ohe_num_classes)
        actions1_ohe = F.one_hot(actions1, num_classes=self.ohe_num_classes)

        # formulate the critic network input
        critic_input = torch.cat((states0, actions0_ohe, states1, actions1_ohe), dim=1)

        # get action primes for critic targets from actor target networks
        actions0_prime = self.get_max_actor_action_target(states0)
        actions1_prime = self.get_max_actor_action_target(states1)

        # one hot encode the action primes for input to the critic target
        actions0_prime_ohe = F.one_hot(actions0_prime, num_classes=self.ohe_num_classes)
        actions1_prime_ohe = F.one_hot(actions1_prime, num_classes=self.ohe_num_classes)

        # formulate the critic target network input
        critic_target_input = torch.cat((state0_primes, actions0_prime_ohe, state1_primes, actions1_prime_ohe), dim=1)

        return critic_input, critic_target_input

    def update_critic(self, critic_q_values, critic_q_targets):
        # back-propagate critic loss and step with the critic optimizer, zero out grads as to not accumulate
        critic_loss = self.loss_function_critic(critic_q_targets, critic_q_values)
        self.optimizer_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
        self.optimizer_critic.step()

    def update_actor(self, critic_q_values, critic_q_targets, states0, actions0, states1, actions1):
        # calculate td_error, needs to critic_q_values copy as to keep computational graphs separate
        td_errors = critic_q_targets - critic_q_values

        # get log probs (range of states pulls each sample, where actions indexes and pulls the correct probability)
        log_probs0 = torch.log(self.actor(states0)[range(len(states0)), actions0]) + 1e-9
        log_probs1 = torch.log(self.actor(states1)[range(len(states1)), actions1]) + 1e-9

        # calculate loss and back-propagate loss and step with optimizer
        actor_loss = (-(log_probs0 * td_errors).mean() - (log_probs1 * td_errors).mean()) / 2
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        self.optimizer_actor.step()

    def get_max_actor_action_target(self, states):
        # use torch context manager to disable gradient computation to save computations
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.tensor(states, dtype=torch.float32)

            if states.dim() == 1:
                states = states.unsqueeze(0)

            action_probs = self.actor_targ(states)

        _, max_action_inds = torch.max(action_probs, dim=1)
        return max_action_inds

    def sample_actor_action_target(self, states):
        # use torch context manager to disable gradient computation to save computations
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.tensor(states, dtype=torch.float32)

            if states.dim() == 1:
                states = states.unsqueeze(0)

            action_probs = self.actor_targ(states)

        action_inds = torch.multinomial(action_probs, num_samples=1).squeeze()
        return action_inds

    def sample_action_from_actor_online(self, state):
        # use torch context manager to disable gradient computation to save computations
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)

            # if state is not a single dimension, throw an error
            if state.dim() != 1:
                raise Exception("State passed to sample action from actor method is greater than one dimension")

            # get the action probabilities
            action_probs = self.actor(state)

            # sample action from the probability distribution (item extracts the number from the tensor)
            action = torch.multinomial(action_probs, num_samples=1).item()

        return action

    # select next action method allows for sampling and exploration to be abstracted away
    def select_next_action(self, state, env):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = self.sample_action_from_actor_online(state)
        return action

    # define an exploration function at compile time, abstract away the decaying
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        # self.epsilon = max(self.epsilon_min, self.epsilon - .00175)
