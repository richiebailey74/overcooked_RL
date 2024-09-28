import gym
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from MAPPO.Visualizer import StudentAgent, StudentPolicy, AgentPair, AgentEvaluator
from DQN import Solver as dqn_Solver, DeepQLearningAgent
from MADDPG import Solver as ctde_solver, CTDEAgent
from MAPPO import Solver as mappo_solver, MAPPOAgent
from MAPPO.Monitoring import visualize_all_graphs
from PIL import Image
import os
from IPython.display import display, Image as IPImage
import dill


if __name__ == '__main__':
    # different layout options
    layout = "cramped_room"  # solved using DQN in 1438 episodes
    # layout = "asymmetric_advantages"  # solved using DQN in 1700 episodes
    # layout = "coordination_ring"
    # layout = "forced_coordination"
    # layout = "counter_circuit_o_1order"

    # need to correspond to the reward values checked in the solver shape_rewards() method
    reward_shaping = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 4,
        "SOUP_PICKUP_REWARD": 5
    }

    horizon = 400  # this is hard coded and the value cannot change
    mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)  # can be one of the five layouts that we need to solve
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    # the above env is passed to solver and used to define the agent's

    control_flow_model = 0
    if control_flow_model == 0:
        layouts = [
            "cramped_room",
            "asymmetric_advantages",
            "coordination_ring",
            "forced_coordination",
            "counter_circuit_o_1order"
        ]
        for lay in layouts:
            dqla = DeepQLearningAgent(
                env=env,
                buffer_cap=100000,
                gamma=.99,
                epsilon=1,
                epsilon_decay=.997,
                epsilon_min=.001,
                learning_rate=.00005,
                batch_size=256
            )
            dqn_solver = dqn_Solver(
                dqla,
                10000,
                env,
                25,
                lay
            )
            dqn_solver.solve(verbose=True)
            dqn_solver.visualize_training_rewards()
            dqn_solver.validate(verbose=True)
            dqn_solver.visualize_validation_rewards()

    elif control_flow_model == 1:
        ctdea = CTDEAgent(
            env=env,
            buffer_cap=200000,
            gamma=.99,
            epsilon=1,
            epsilon_decay=.997,
            epsilon_min=.001,
            learning_rate=.00005,
            batch_size=256
        )
        ctdesolver = ctde_solver(
            ctdea,
            10000,
            env,
            15,
            layout,
            True
        )
        ctdesolver.solve(verbose=True)
        ctdesolver.visualize_training_rewards()

    elif control_flow_model == 2:
        visualize_model_trajs = True
        try:
            mappoa = MAPPOAgent(
                env=env,
                buffer_cap=400,
                gamma=.99,
                learning_rate_actor=.0001,  # TODO: try even higher learning rates here
                learning_rate_critic=.0003,
                epsilon_clip=.08,  # heuristically should be in between .1 and .2
                lambda_gae=.95,  # heuristically should be in between .9 and 1.0
                batch_size=200
            )
            mapposolver = mappo_solver(
                mappoa,
                2000,
                env,
                10,
                layout,
            )
            mapposolver.solve(verbose=True)
            mapposolver.visualize_training_rewards()

            if visualize_model_trajs:
                # Instantiate the policies for both agents
                policy0 = StudentPolicy(env, mappoa)
                policy1 = StudentPolicy(env, mappoa)

                # Instantiate both agents
                agent0 = StudentAgent(policy0)
                agent1 = StudentAgent(policy1)
                agent_pair = AgentPair(agent0, agent1)

                # Generate an episode
                ae = AgentEvaluator.from_layout_name({"layout_name": layout}, {"horizon": horizon})
                trajs = ae.evaluate_agent_pair(agent_pair, num_games=1)
                print("\nlen(trajs):", len(trajs))
                img_dir = "imgs/"  # "/content/drive/My Drive/Colab/" + "imgs_" + layout + "/"
                ipython_display = True
                gif_path = "./imgs.gif"  # "/content/drive/My Drive/Colab/" + layout + ".gif"

                StateVisualizer().display_rendered_trajectory(trajs, img_directory_path=img_dir,
                                                              ipython_display=ipython_display)

                img_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
                img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
                images = [Image.open(img_dir + img).convert('RGBA') for img in img_list]
                images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=250, loop=0)
                with open(gif_path, 'rb') as f: display(IPImage(data=f.read(), format='png'))

                visualize_all_graphs()
                StateVisualizer()

        except KeyboardInterrupt:
            visualize_all_graphs()
            StateVisualizer()
