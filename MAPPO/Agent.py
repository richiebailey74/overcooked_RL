from .Networks import DeepActorNetwork, DeepCriticNetwork, DeepRandomDistillationNetwork
import torch
from .ExperienceBuffer import ExperienceBuffer
from .Monitoring import (
    loss_monitoring_total,
    loss_monitoring_critic,
    loss_monitoring_actor0,
    loss_monitoring_actor1,
    weight_monitoring_critic,
    gradient_monitoring_critic,
    weight_monitoring_actor0,
    gradient_monitoring_actor0,
    weight_monitoring_actor1,
    gradient_monitoring_actor1
)
import copy
import numpy as np


def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)



class MAPPOAgent:
    def __init__(
            self,
            env,
            buffer_cap,
            gamma,
            learning_rate_actor,
            learning_rate_critic,
            epsilon_clip,
            lambda_gae,
            batch_size=1,
            loss_function=torch.nn.MSELoss()
    ):
        self.actor0 = None
        self.actor1 = None
        self.critic = None
        self.rnd_target = None
        self.rnd_predictor = None
        self.define_deep_nets(env.observation_space.shape[0], env.action_space.n)
        self.ohe_num_classes = env.action_space.n

        self.experience_buffer = None
        self.buffer_cap = buffer_cap
        self.batch_size = batch_size
        self.obs_space_len = env.observation_space.shape[0]
        self.reset_experience_buffer()

        self.gamma = gamma
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.epsilon_clip = epsilon_clip
        self.lambda_gae = lambda_gae
        self.loss_function_mse = loss_function
        self.optimizer_actor0 = torch.optim.NAdam(self.actor0.parameters(), lr=learning_rate_actor)
        self.optimizer_actor1 = torch.optim.NAdam(self.actor1.parameters(), lr=learning_rate_actor)
        self.optimizer_critic = torch.optim.NAdam(self.critic.parameters(), lr=learning_rate_critic)
        self.optimizer_rnd_predictor = torch.optim.NAdam(self.rnd_predictor.parameters(), lr=learning_rate_actor)

    def define_deep_nets(self, state_dimensionality, action_dimensionality):
        self.actor0 = DeepActorNetwork(state_dimensionality, action_dimensionality)
        self.actor1 = DeepActorNetwork(state_dimensionality, action_dimensionality)
        self.critic = DeepCriticNetwork(state_dimensionality + 4, 1)
        self.rnd_predictor = DeepRandomDistillationNetwork(state_dimensionality, state_dimensionality)
        self.rnd_target = DeepRandomDistillationNetwork(state_dimensionality, state_dimensionality)

        self.critic.apply(initialize_weights)
        self.actor0.apply(initialize_weights)
        self.actor1.apply(initialize_weights)
        self.rnd_predictor.apply(initialize_weights)
        self.rnd_target.apply(initialize_weights)

        # freeze parameters for RND target network
        for param in self.rnd_target.parameters():
            param.requires_grad = False

    def reset_experience_buffer(self):
        self.experience_buffer = ExperienceBuffer(self.buffer_cap, self.batch_size, self.obs_space_len)

    def update_actor_critic_networks(self):
        if self.experience_buffer.advantages is None:
            raise Exception("Advantages have not been set, error in training loop in setting values")
        # sample experience buffer
        (states0, actions0, states1, actions1, rewards, shaped_rewards0, shaped_rewards1, state0_primes, state1_primes,
         critic_values_prime, action_probs_old0, action_probs_old1, advantages, critic_losses, terminations) = (
            self.experience_buffer.sample())

        critic_input, critic_input_prime = self.get_critic_inputs(states0, states1, state0_primes, state1_primes)

        # get V values from critic and critic target
        critic_v_values = self.critic(critic_input)
        # print("Critic values shape", critic_v_values.size())

        # recalculate the primes
        critic_values_prime = self.critic(critic_input_prime)

        # construct the v targets (unsqueeze rewards to make dimensions compatible)
        critic_v_targets = (rewards + shaped_rewards0 + shaped_rewards1).unsqueeze(-1) + (
                self.gamma * critic_values_prime)

        # calculate critic loss
        # loss_critic = self.loss_function_mse(critic_v_values, critic_v_targets)
        loss_critic = (critic_v_targets - critic_v_values) ** 2

        # calculate the advantages
        advantages = self.calculate_advantages(rewards, critic_v_values, terminations)

        # calculate critic loss
        loss_critic = torch.mean(loss_critic)

        # generate new policy action probabilities
        action_probs_new0, action_probs_new1 = self.generate_action_probs(states0, states1, actions0, actions1, False)

        # make sure no division or logs by zero occurs, divisions are always dangerous, then re-normalize
        epsilon = 1e-12

        # get rid of zeros as to make numerically sound
        zero_mask = (action_probs_old0 == 0)
        action_probs_old0[zero_mask] = epsilon
        action_probs_old0 = torch.log(action_probs_old0 / action_probs_old0.sum())
        zero_mask = (action_probs_old1 == 0)
        action_probs_old1[zero_mask] = epsilon
        action_probs_old1 = torch.log(action_probs_old1 / action_probs_old1.sum())

        zero_mask = (action_probs_new0 == 0)
        action_probs_new0[zero_mask] = epsilon
        action_probs_new0 = torch.log(action_probs_new0 / action_probs_new0.sum())
        zero_mask = (action_probs_new1 == 0)
        action_probs_new1[zero_mask] = epsilon
        action_probs_new1 = torch.log(action_probs_new1 / action_probs_new1.sum())

        # generate probability ratio and clipping it with epsilon clip
        probability_ratios0 = torch.exp(action_probs_new0) / torch.exp(action_probs_old0)
        probability_ratios1 = torch.exp(action_probs_new1) / torch.exp(action_probs_old1)
        clipped_ratios0 = torch.clamp(probability_ratios0, min=1 - self.epsilon_clip, max=1 + self.epsilon_clip)
        clipped_ratios1 = torch.clamp(probability_ratios1, min=1 - self.epsilon_clip, max=1 + self.epsilon_clip)

        # get objectives and use them to calculate the clipped objective and the loss
        unclipped_objective0 = probability_ratios0 * advantages
        clipped_objective0 = clipped_ratios0 * advantages
        surrogate_objective0 = torch.min(unclipped_objective0, clipped_objective0)
        loss_actor0 = -torch.mean(surrogate_objective0)

        unclipped_objective1 = probability_ratios1 * advantages
        clipped_objective1 = clipped_ratios1 * advantages
        surrogate_objective1 = torch.min(unclipped_objective1, clipped_objective1)
        loss_actor1 = -torch.mean(surrogate_objective1)

        # update the neural networks with the associated losses
        combined_loss = (loss_actor0 + loss_actor1 + (.5 * loss_critic))
        self.optimizer_actor0.zero_grad()
        self.optimizer_actor1.zero_grad()
        self.optimizer_critic.zero_grad()
        # print("Shape of combined loss", combined_loss.size())
        combined_loss.backward()  # propagates though all the neural nets attached to the computation graph

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(self.actor0.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(self.actor1.parameters(), max_norm=1)
        # You can adjust `max_norm` to a suitable value based on your observations or experimentally.

        self.optimizer_actor0.step()
        self.optimizer_actor1.step()
        self.optimizer_critic.step()

        # self.optimizer_actor0.zero_grad()
        # loss_actor0.backward(retain_graph=True)
        # self.optimizer_actor0.step()
        #
        # self.optimizer_actor1.zero_grad()
        # loss_actor1.backward(retain_graph=True)
        # self.optimizer_actor1.step()
        #
        # self.optimizer_critic.zero_grad()
        # loss_critic.backward()
        # self.optimizer_critic.step()

        loss_monitoring_total.append(combined_loss.detach().numpy())
        loss_monitoring_critic.append(loss_critic.detach().numpy())
        loss_monitoring_actor0.append(loss_actor0.detach().numpy())
        loss_monitoring_actor1.append(loss_actor1.detach().numpy())

        for name, parameter in self.critic.named_parameters():
            if parameter.grad is not None:
                grad_norm = parameter.grad.norm()
                gradient_monitoring_critic[name].append(grad_norm)

        for name, parameter in self.actor0.named_parameters():
            if parameter.grad is not None:
                grad_norm = parameter.grad.norm()
                gradient_monitoring_actor0[name].append(grad_norm)

        for name, parameter in self.actor1.named_parameters():
            if parameter.grad is not None:
                grad_norm = parameter.grad.norm()
                gradient_monitoring_actor1[name].append(grad_norm)

        for name, parameter in self.critic.named_parameters():
            if parameter.requires_grad:  # Ensuring only trainable parameters are considered
                weight_monitoring_critic[name]["norm"].append(parameter.data.norm())
                weight_monitoring_critic[name]["mean"].append(parameter.data.mean())
                weight_monitoring_critic[name]["std"].append(parameter.data.std())
                weight_monitoring_critic[name]["max"].append(parameter.data.max())
                weight_monitoring_critic[name]["min"].append(parameter.data.min())

        for name, parameter in self.actor0.named_parameters():
            if parameter.requires_grad:  # Ensuring only trainable parameters are considered
                weight_monitoring_actor0[name]["norm"].append(parameter.data.norm())
                weight_monitoring_actor0[name]["mean"].append(parameter.data.mean())
                weight_monitoring_actor0[name]["std"].append(parameter.data.std())
                weight_monitoring_actor0[name]["max"].append(parameter.data.max())
                weight_monitoring_actor0[name]["min"].append(parameter.data.min())

        for name, parameter in self.actor1.named_parameters():
            if parameter.requires_grad:  # Ensuring only trainable parameters are considered
                weight_monitoring_actor1[name]["norm"].append(parameter.data.norm())
                weight_monitoring_actor1[name]["mean"].append(parameter.data.mean())
                weight_monitoring_actor1[name]["std"].append(parameter.data.std())
                weight_monitoring_actor1[name]["max"].append(parameter.data.max())
                weight_monitoring_actor1[name]["min"].append(parameter.data.min())

    def remove_redundant_state_information_for_critic(self, states):
        column_indices_to_keep = [92, 93, 94, 95]
        return states[:, column_indices_to_keep]

    def generate_intrinsic_reward(self, state0, state1):
        self.rnd_target.eval()
        self.rnd_predictor.eval()

        target0 = self.rnd_target(state0)
        prediction0 = self.rnd_predictor(state0)
        target1 = self.rnd_target(state1)
        prediction1 = self.rnd_predictor(state1)
        intrinsic_rew = self.loss_function_mse(target0, prediction0).item() + self.loss_function_mse(target1, prediction1).item()
        return intrinsic_rew

    def update_rnd_predictor_network(self):
        states0, states1 = self.experience_buffer.get_all_states()
        self.rnd_predictor.train()  # Ensure the network is in training mode

        # Get corresponding target network outputs
        self.rnd_target.eval()  # Ensure the target network is in eval mode
        with torch.no_grad():
            target_outputs0 = self.rnd_target(states0)
            target_outputs1 = self.rnd_target(states1)

        # Get predictor outputs
        predictor_outputs0 = self.rnd_predictor(states0)
        predictor_outputs1 = self.rnd_predictor(states1)

        # Calculate loss
        loss0 = self.loss_function_mse(predictor_outputs0, target_outputs0)
        loss1 = self.loss_function_mse(predictor_outputs1, target_outputs1)

        # Backpropagation
        self.optimizer_rnd_predictor.zero_grad()
        (loss0 + loss1).backward()
        self.optimizer_rnd_predictor.step()

    # critic network is a V network so no action inputs needed
    def get_critic_inputs(self, states0, states1, state0_primes, state1_primes):

        # formulate the critic network input
        critic_input = torch.cat((
            states0,
            self.remove_redundant_state_information_for_critic(states1),
        ), dim=1)

        # formulate the critic target network input
        critic_target_input = torch.cat((
            state0_primes,
            self.remove_redundant_state_information_for_critic(state1_primes),
        ), dim=1)

        return critic_input, critic_target_input

    def set_advantages_and_critic_loss(self):
        (states0, actions0, states1, actions1, rewards, shaped_rewards0, shaped_rewards1, state_primes0, state_primes1,
         critic_values_prime, action_probs0, action_probs1, terminations) = (
            self.experience_buffer.get_entire_buffer_for_advantages_critic_losses())
        # get critic inputs
        critic_input, critic_input_prime = self.get_critic_inputs(states0, states1, state_primes0, state_primes1)

        # get V values from critic and critic target
        critic_v_values = self.critic(critic_input)
        # print("Critic values shape", critic_v_values.size())

        # recalculate the primes
        critic_values_prime = self.critic(critic_input_prime)

        # TODO: add random network distillation here as more rewards for learning and encouraging exploration
        # construct the v targets (unsqueeze rewards to make dimensions compatible)
        critic_v_targets = (rewards + shaped_rewards0 + shaped_rewards1).unsqueeze(-1) + (
                    self.gamma * critic_values_prime)

        # calculate critic loss
        # loss_critic = self.loss_function_mse(critic_v_values, critic_v_targets)
        loss_critic = (critic_v_targets - critic_v_values) ** 2

        # calculate the advantages
        advantages = self.calculate_advantages(rewards, critic_v_values, terminations)

        # set the advantages and critic losses in the experience buffer so the update method can efficiently sample
        self.experience_buffer.set_advantages_and_critic_loss(advantages.detach().numpy(), loss_critic.detach().numpy())

    # def calculate_advantages(self, td_errors, rewards, values, terminated):
    #     advantages = torch.zeros_like(td_errors)
    #     for t in range(400):
    #         a_t = 0
    #         discount = 1
    #         for k in reversed(range(td_errors.size(0) - 1)):
    #             a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - int(terminated[k])) - values[k])
    #             discount *= self.gamma * self.lambda_gae
    #         advantages[t] = a_t
    #
    #     return advantages

    def calculate_advantages(self, rewards, values, terminated):
        n = rewards.size(0)  # Total number of timesteps
        advantages = torch.zeros_like(rewards)
        last_advantage = 0  # Last advantage initialization

        # We calculate in reverse from the last timestep to the first
        for t in reversed(range(n)):
            if t == n - 1 or terminated[t]:
                delta = rewards[t] - values[t]  # No next state if it's the last timestep or terminated
                last_advantage = delta  # Advantage is just the delta at the end of the episode
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                last_advantage = delta + self.gamma * self.lambda_gae * last_advantage

            advantages[t] = last_advantage

        return advantages


    # def sample_actors_action_target(self, states0, states1):
    #     # use torch context manager to disable gradient computation to save computations
    #     with torch.no_grad():
    #         if not isinstance(states0, torch.Tensor):
    #             states0 = torch.tensor(states0, dtype=torch.float32)
    #         if not isinstance(states1, torch.Tensor):
    #             states1 = torch.tensor(states1, dtype=torch.float32)
    #
    #         if states0.dim() == 1:
    #             states0 = states0.unsqueeze(0)
    #         if states1.dim() == 1:
    #             states1 = states1.unsqueeze(0)
    #
    #         action_probs0 = self.actor_targ0(states0)
    #         action_probs1 = self.actor_targ1(states1)
    #
    #     action_inds0 = torch.multinomial(action_probs0, num_samples=1).squeeze()
    #     action_inds1 = torch.multinomial(action_probs1, num_samples=1).squeeze()
    #     return action_inds0, action_inds1

    def generate_action_probs(self, states0, states1, actions0, actions1, no_grad=True):
        if no_grad:
            with torch.no_grad():
                if not isinstance(states0, torch.Tensor):
                    states0 = torch.tensor(states0, dtype=torch.float32)
                if not isinstance(states1, torch.Tensor):
                    states1 = torch.tensor(states1, dtype=torch.float32)
                if not isinstance(actions0, torch.Tensor):
                    actions0 = torch.tensor(actions0, dtype=torch.int64)
                if not isinstance(actions1, torch.Tensor):
                    actions1 = torch.tensor(actions1, dtype=torch.int64)

                if states0.dim() == 1:
                    states0 = states0.unsqueeze(0)
                if states1.dim() == 1:
                    states1 = states1.unsqueeze(0)
                if actions0.dim() == 1:
                    actions0 = actions0.unsqueeze(0)
                elif actions0.dim() == 0:
                    actions0 = actions0.unsqueeze(0).unsqueeze(1)
                if actions1.dim() == 1:
                    actions1 = actions1.unsqueeze(0)
                elif actions1.dim() == 0:
                    actions1 = actions1.unsqueeze(0).unsqueeze(1)
        else:
            if not isinstance(states0, torch.Tensor):
                states0 = torch.tensor(states0, dtype=torch.float32)
            if not isinstance(states1, torch.Tensor):
                states1 = torch.tensor(states1, dtype=torch.float32)
            if not isinstance(actions0, torch.Tensor):
                actions0 = torch.tensor(actions0, dtype=torch.int64)
            if not isinstance(actions1, torch.Tensor):
                actions1 = torch.tensor(actions1, dtype=torch.int64)

            if states0.dim() == 1:
                states0 = states0.unsqueeze(0)
            if states1.dim() == 1:
                states1 = states1.unsqueeze(0)
            if actions0.dim() == 1:
                actions0 = actions0.unsqueeze(0)
            elif actions0.dim() == 0:
                actions0 = actions0.unsqueeze(0).unsqueeze(1)
            if actions1.dim() == 1:
                actions1 = actions1.unsqueeze(0)
            elif actions1.dim() == 0:
                actions1 = actions1.unsqueeze(0).unsqueeze(1)

        # call the actor networks to get the action probabilities
        action_probs0 = self.actor0(states0)
        action_probs1 = self.actor1(states1)

        # use actions0/1 as indices to extract the relevant probabilities
        indexed_action_probs0 = torch.gather(action_probs0, 1, actions0.transpose(0, 1)).squeeze(1)
        indexed_action_probs1 = torch.gather(action_probs1, 1, actions1.transpose(0, 1)).squeeze(1)

        return indexed_action_probs0, indexed_action_probs1

    def sample_actors_single_action_online(self, state0, state1):
        # use torch context manager to disable gradient computation to save computations
        with torch.no_grad():
            if not isinstance(state0, torch.Tensor):
                state0 = torch.tensor(state0, dtype=torch.float32)
            if not isinstance(state1, torch.Tensor):
                state1 = torch.tensor(state1, dtype=torch.float32)

            # if state is not a single dimension, throw an error
            if state0.dim() != 1:
                raise Exception("State passed to sample action from actor method is greater than one dimension")
            if state1.dim() != 1:
                raise Exception("State passed to sample action from actor method is greater than one dimension")

            # get the action probabilities
            action_probs0 = self.actor0(state0)
            action_probs1 = self.actor1(state1)

            # print("action probs 0", action_probs0)
            # print("action probs 1", action_probs1)

            # sample action from the probability distribution (item extracts the number from the tensor)
            action0 = torch.multinomial(action_probs0, num_samples=1).item()
            action1 = torch.multinomial(action_probs1, num_samples=1).item()

        return action0, action1

    # select next action method allows for sampling and exploration to be abstracted away
    def select_next_action(self, state0, state1):
        action0, action1 = self.sample_actors_single_action_online(state0, state1)
        return action0, action1
