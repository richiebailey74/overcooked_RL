import torch
import matplotlib.pyplot as plt
import numpy as np


class Solver:

    def __init__(self, agent, max_episodes, env, tna_c, layout):
        self.agent = agent

        self.env = env
        self.layout = layout

        self.max_episodes = max_episodes
        self.tna_c = tna_c  # realign target network after this many steps

        self.reward_array = []
        self.val_rew_array = []

    def solve(self, verbose=False):
        total_it = 0
        for ep_count in range(self.max_episodes):
            state = self.env.reset()
            s0 = state["both_agent_obs"][0]
            s1 = state["both_agent_obs"][1]
            action0 = self.agent.select_action_online(s0, self.env)
            action1 = self.agent.select_action_online(s1, self.env)
            cumulative_reward = 0
            terminate_episode = False
            soups_made = 0
            it = 0
            while not terminate_episode:
                action0 = action0.item() if isinstance(action0, torch.Tensor) else action0
                action1 = action1.item() if isinstance(action1, torch.Tensor) else action1
                state_prime, reward, terminate_episode, info = self.env.step([action0, action1])
                sp0 = state_prime["both_agent_obs"][0]
                sp1 = state_prime["both_agent_obs"][1]

                r_shaped_0, r_shaped_1 = self.shape_rewards(s0, s1, reward, sp0, sp1)
                if self.env.agent_idx:
                    r_shaped_0 = r_shaped_1
                    r_shaped_1 = r_shaped_0
                else:
                    r_shaped_0 = r_shaped_0
                    r_shaped_1 = r_shaped_1

                # don't need an early terminating condition since horizon parameter passed to the MDP

                self.agent.experience_buffer.add(s0, action0, reward + r_shaped_0, sp0, terminate_episode)
                self.agent.experience_buffer.add(s1, action1, reward + r_shaped_1, sp1, terminate_episode)

                self.agent.update_q_online_network()
                self.agent.update_q_online_network()

                action_prime0 = self.agent.select_action_online(sp0, self.env)
                action_prime1 = self.agent.select_action_online(sp1, self.env)

                s0, s1, action0, action1 = sp0, sp1, action_prime0, action_prime1

                cumulative_reward += reward
                soups_made += int(reward / 20)

                it += 1
                total_it += 1

                # since C divides iteration number, update the target Q network's values
                if total_it % self.tna_c == 0:
                    self.agent.update_q_target_network()

            # add cumulative reward to the array for visualization
            self.reward_array.append(cumulative_reward)

            self.agent.decay_epsilon()

            if verbose:
                avg_soup = np.mean(self.reward_array) / 20
                print(f"Episode {ep_count} completed with {cumulative_reward // 20} soups "
                      f"and {avg_soup} average soups")
                if avg_soup >= 7:
                    print("Daily average soups exceed 7 over the every episode")
                    return

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
        moving_average = list(map(lambda i: sum(self.reward_array[max(0, i-19):i+1]) / len(self.reward_array[max(0, i-19):i+1]), range(len(self.reward_array))))

        plt.plot(list(range(len(moving_average))), moving_average, label=f"{self.layout} Reward")
        plt.title("Overcooked Deep RL: Reward vs Episodes\nfor Training DQL for Each Agent")
        # plt.ylim(bottom=-400)
        # plt.axhline(y=200, color='#800000', linestyle='--')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(f"figures/dqn_reward_curve_training_{self.layout}.png")
        plt.show()

    def validate(self, verbose=False):
        for ep_count in range(100):
            state = self.env.reset()
            s0 = state["both_agent_obs"][0]
            s1 = state["both_agent_obs"][1]
            action0 = self.agent.select_action_online(s0, self.env)
            action1 = self.agent.select_action_online(s1, self.env)
            cumulative_reward = 0
            terminate_episode = False
            soups_made = 0
            while not terminate_episode:
                action0 = action0.item() if isinstance(action0, torch.Tensor) else action0
                action1 = action1.item() if isinstance(action1, torch.Tensor) else action1
                state_prime, reward, terminate_episode, info = self.env.step([action0, action1])
                sp0 = state_prime["both_agent_obs"][0]
                sp1 = state_prime["both_agent_obs"][1]

                action_prime0 = self.agent.select_action_online(sp0, self.env)
                action_prime1 = self.agent.select_action_online(sp1, self.env)

                s0, s1, action0, action1 = sp0, sp1, action_prime0, action_prime1

                cumulative_reward += reward
                soups_made += int(reward / 20)

            # add cumulative reward to the array for visualization
            self.val_rew_array.append(cumulative_reward)

            if verbose:
                avg_soup = np.mean(self.val_rew_array) / 20
                print(f"Validation episode {ep_count} completed with {cumulative_reward // 20} soups "
                      f"and {avg_soup} average soups")

    def visualize_validation_rewards(self):
        plt.plot(list(range(len(self.val_rew_array))), self.val_rew_array, label=f"{self.layout} Reward")
        plt.title("Overcooked Deep RL: Reward vs Episodes\nfor Validation DQL for Each Agent")
        # plt.ylim(bottom=-400)
        # plt.axhline(y=200, color='#800000', linestyle='--')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(f"figures/dqn_reward_curve_validation_{self.layout}.png")
        plt.show()
