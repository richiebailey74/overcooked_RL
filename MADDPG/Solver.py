import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pygame
from matplotlib.animation import FuncAnimation
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from PIL import Image
from IPython.display import display, Image as IPImage
import time as time
import os
import subprocess
import imageio


class Solver:
    def __init__(self, agent, max_episodes, env, tna_c, layout, use_visualizer):
        self.agent = agent

        self.env = env
        self.layout = layout

        self.max_episodes = max_episodes
        self.tna_c = tna_c  # realign target network after this many steps

        self.reward_array = []
        self.visualizer = StateVisualizer()
        self.use_visualizer = use_visualizer

    def solve(self, verbose=False):
        if self.use_visualizer:
            pygame.init()
            screen = pygame.display.set_mode((640, 480))
        total_it = 0
        for ep_count in range(self.max_episodes):
            state = self.env.reset()
            s0 = state["both_agent_obs"][0]
            s1 = state["both_agent_obs"][1]
            action0 = self.agent.select_next_action(s0, self.env)
            action1 = self.agent.select_next_action(s1, self.env)
            cumulative_reward = 0
            terminate_episode = False
            soups_made = 0
            it = 0
            images_to_display = []
            while not terminate_episode:
                action0 = action0.item() if isinstance(action0, torch.Tensor) else action0
                action1 = action1.item() if isinstance(action1, torch.Tensor) else action1
                state_prime, reward, terminate_episode, info = self.env.step([action0, action1])
                sp0 = state_prime["both_agent_obs"][0]
                sp1 = state_prime["both_agent_obs"][1]

                # if self.use_visualizer == True:
                #     viz = self.visualizer.display_rendered_state(state=self.env.base_env.state, grid=self.env.mdp.terrain_mtx, ipython_display=True)
                #     # display(IPImage(filename=viz))
                #     # if os.path.exists(viz):
                #     #     subprocess.run(
                #     #         ['open', viz])  # macOS and Linux (with 'xdg-open' instead of 'open' for Linux)
                #     #     # For Windows, use subprocess.run(['start', image_path], shell=True)
                #     # else:
                #     #     print(f"File not found: {viz}")
                #     # img = mpimg.imread(viz)
                #     # plt.imshow(img)
                #     # plt.axis('off')  # Hide axes
                #     # plt.show()
                #     # time.sleep(.1)
                #     images_to_display.append(viz)


                # r_shaped = info["shaped_r_by_agent"]
                # if self.env.agent_idx:
                #     r_shaped_0 = r_shaped[1]
                #     r_shaped_1 = r_shaped[0]
                # else:
                #     r_shaped_0 = r_shaped[0]
                #     r_shaped_1 = r_shaped[1]
                #
                # r_shaped_0, r_shaped_1 = self.shape_rewards(s0, s1, r_shaped_0, r_shaped_1)

                r_shaped_0, r_shaped_1 = 0, 0

                if r_shaped_0 != 0 or r_shaped_1 != 0:
                    print(f"Reward shaping occurring: r0: {r_shaped_0}, r1: {r_shaped_1}")

                # don't need an early terminating condition since horizon parameter passed to the MDP

                self.agent.experience_buffer.add(s0, action0, s1, action1, reward, r_shaped_0, r_shaped_1, sp0, sp1, terminate_episode)

                self.agent.update_actor_critic_networks()

                action_prime0 = self.agent.select_next_action(sp0, self.env)
                action_prime1 = self.agent.select_next_action(sp1, self.env)

                s0, s1, action0, action1 = sp0, sp1, action_prime0, action_prime1

                cumulative_reward += reward
                soups_made += int(reward / 20)

                it += 1
                total_it += 1

                # since C divides iteration number, update the target Q network's values
                if total_it % self.tna_c == 0:
                    self.agent.update_target_networks()

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

    def shape_rewards(self, s0, s1, sr0, sr1):
        is_holding_dish = 6
        onions_in_closest_soup_ind = 16
        is_soup_ready_ind = 26
        if sr0 == 5:
            if s0[onions_in_closest_soup_ind] >= 3:
                sr0 = -5
        elif sr0 == 4:
            if s0[onions_in_closest_soup_ind] < 3 and s1[onions_in_closest_soup_ind] < 3:
                sr0 = -4
        elif sr0 == 3:
            # print("state", s0)
            if s0[is_soup_ready_ind] == 0 or s0[onions_in_closest_soup_ind] < 3 or s0[is_holding_dish] == 0:
                sr0 = -3

        if sr1 == 5:
            if s1[onions_in_closest_soup_ind] >= 3:
                sr1 = -5
        elif sr1 == 4:
            if s0[onions_in_closest_soup_ind] < 3 and s1[onions_in_closest_soup_ind] < 3:
                sr1 = -4
        elif sr1 == 3:
            # print("state", s1)
            if s1[is_soup_ready_ind] == 0 or s1[onions_in_closest_soup_ind] < 3 or s0[is_holding_dish] == 0:
                sr1 = -3

        return sr0, sr1

    def visualize_training_rewards(self):
        plt.plot(list(range(len(self.reward_array))), self.reward_array, label=f"{self.layout} Reward")
        plt.title("Overcooked Deep RL: Reward vs Episodes\nfor Training CTDE for Each Agent")
        # plt.ylim(bottom=-400)
        # plt.axhline(y=200, color='#800000', linestyle='--')
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(f"../figures/ctde_reward_curve_training_{self.layout}.png")
        plt.show()


