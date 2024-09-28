import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import copy


class Solver:
    def __init__(self, agent, max_episodes, env, epochs_per_update, layout):
        self.agent = agent

        self.env = env
        self.layout = layout

        self.max_episodes = max_episodes
        self.epochs_per_update = epochs_per_update

        self.reward_array = []

    def solve(self, verbose=False):
        total_it = 0
        for ep_count in range(self.max_episodes):
            state = self.env.reset()
            s0 = state["both_agent_obs"][0]
            s1 = state["both_agent_obs"][1]
            action0, action1 = self.agent.select_next_action(s0, s1)
            cumulative_reward = 0
            terminate_episode = False
            soups_made = 0

            # data collection loop
            while not terminate_episode:
                action0 = action0.item() if isinstance(action0, torch.Tensor) else action0
                action1 = action1.item() if isinstance(action1, torch.Tensor) else action1
                state_prime, reward, terminate_episode, info = self.env.step([action0, action1])
                sp0 = state_prime["both_agent_obs"][0]
                sp1 = state_prime["both_agent_obs"][1]

                r_shaped = info["shaped_r_by_agent"]
                r_shaped_0, r_shaped_1 = self.shape_rewards(s0, s1, reward, sp0, sp1)
                if self.env.agent_idx:
                    r_shaped_0 = r_shaped_1
                    r_shaped_1 = r_shaped_0
                else:
                    r_shaped_0 = r_shaped_0
                    r_shaped_1 = r_shaped_1

                # don't need an early terminating condition since horizon parameter passed to the MDP

                # get action probabilities to store in experience buffer so clipped ratio can be calculated in updating
                action_probs0, action_probs1 = self.agent.generate_action_probs(s0, s1, action0, action1)

                # get critic estimate for current policy for the next state to store in the experience buffer
                _, critic_input_prime = self.agent.get_critic_inputs(
                    torch.from_numpy(s0).unsqueeze(0).float(),
                    torch.from_numpy(s1).unsqueeze(0).float(),
                    torch.from_numpy(sp0).unsqueeze(0).float(),
                    torch.from_numpy(sp1).unsqueeze(0).float()
                )
                critic_value_prime = self.agent.critic(critic_input_prime)

                intrinsic_reward = self.agent.generate_intrinsic_reward(
                    torch.from_numpy(s0).unsqueeze(0).float(),
                    torch.from_numpy(s1).unsqueeze(0).float()
                )

                self.agent.experience_buffer.add(
                    s0,
                    action0,
                    s1,
                    action1,
                    reward + intrinsic_reward,
                    r_shaped_0,
                    r_shaped_1,
                    sp0,
                    sp1,
                    critic_value_prime.item(),
                    copy(action_probs0).detach().numpy(),
                    copy(action_probs1).detach().numpy(),
                    terminate_episode
                )

                action_prime0, action_prime1 = self.agent.select_next_action(sp0, sp1)

                s0, s1, action0, action1 = sp0, sp1, action_prime0, action_prime1

                cumulative_reward += reward
                soups_made += int(reward / 20)

            # add cumulative reward to the array for visualization
            self.reward_array.append(cumulative_reward)

            if verbose:
                avg_soup = np.mean(self.reward_array) / 20
                print(f"Episode {ep_count} completed with {cumulative_reward // 20} soups "
                      f"and {avg_soup} average soups")
                if avg_soup >= 7:
                    print("Daily average soups exceed 7 over the every episode")
                    return

            # set advantages and critic losses for all time steps in this episode so batch calculations are efficient
            self.agent.set_advantages_and_critic_loss()

            # if total_it % 10 == 0:
            # learning loop to learn over the epochs hyperparameter
            for _ in range(self.epochs_per_update):
                # perform a single update to the actor critic
                self.agent.update_actor_critic_networks()

            self.agent.update_rnd_predictor_network()

            self.agent.reset_experience_buffer()

            total_it += 1

    def get_shaped_reward_agent(self, s, sp, reward):
        shaped_reward = 0
        is_holding_onion = 4
        is_holding_soup = 5
        is_holding_dish = 6
        onions_in_closest_soup = 16
        is_soup_full = 24
        is_soup_cooking = 25
        is_soup_ready = 26
        is_holding_nothing = s[is_holding_onion] == 0 and s[is_holding_soup] == 0 and s[is_holding_dish] == 0
        # segment out based on actions (state changes) and from there dictate states of things to control reward
        if s[is_holding_soup] == 0 and sp[is_holding_soup] == 1:
            # picks up soup, check if it was ready and full
            if s[is_soup_full] == 1 and s[is_soup_ready] == 1:
                shaped_reward += 2
            else:
                if s[onions_in_closest_soup] == 0:
                    shaped_reward -= .025
                elif s[onions_in_closest_soup] == 1:
                    shaped_reward -= .05
                elif s[onions_in_closest_soup] == 2:
                    shaped_reward -= .075
                elif s[onions_in_closest_soup] >= 3:
                    shaped_reward -= .1
            pass
        elif s[is_holding_onion] == 0 and sp[is_holding_onion] == 1:
            # picks up onion, check if any are needed and reward accordingly
            if s[onions_in_closest_soup] < 3:
                shaped_reward += .01
            else:
                shaped_reward -= .1
        elif s[is_holding_dish] == 0 and sp[is_holding_dish] == 1:
            # picks up dish
            shaped_reward += .01
        elif s[is_holding_soup] == 1 and sp[is_holding_soup] == 0:
            # drops soup, check if it was a successful delivery
            if reward != 20:
                shaped_reward -= 3
        elif s[is_holding_onion] == 1 and sp[is_holding_onion] == 0:
            # drops onion, see if in pot
            if s[onions_in_closest_soup] == 0 and sp[onions_in_closest_soup] == 1:
                shaped_reward += .025
            elif s[onions_in_closest_soup] == 1 and sp[onions_in_closest_soup] == 2:
                shaped_reward += .05
            elif s[onions_in_closest_soup] == 2 and sp[onions_in_closest_soup] == 3:
                shaped_reward += .075
            else:
                shaped_reward -= .01
        elif s[is_holding_dish] == 1 and sp[is_holding_dish] == 0:
            # drops dish, see where
            shaped_reward -= .01
        elif s[is_soup_cooking] == 0 and sp[is_soup_cooking] == 1:
            # started cooking the soup in pot, make sure it's full otherwise penalize
            if s[is_soup_full] == 1:
                shaped_reward += .25
            else:
                shaped_reward -= .25
        elif s[is_soup_cooking] == 1 and s[is_soup_full] == 1:
            shaped_reward = 0
        else:
            shaped_reward -= .0025

        return shaped_reward


    def shape_rewards(self, s0, s1, reward, sp0, sp1):

        sr0 = self.get_shaped_reward_agent(s0, sp0, reward)
        sr1 = self.get_shaped_reward_agent(s1, sp1, reward)

        return sr0, sr1

    def visualize_training_rewards(self):
        plt.plot(list(range(len(self.reward_array))), self.reward_array, label=f"{self.layout} Reward")
        plt.title("Overcooked Deep RL: Reward vs Episodes\nfor Training CTDE for Each Agent")
        # plt.ylim(bottom=-400)
        # plt.axhline(y=200, color='#800000', linestyle='--')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(f"figures/mappo_reward_curve_training_{self.layout}.png")
        plt.show()


