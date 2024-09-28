from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import gym
import numpy as np
import torch
from PIL import Image
import os
from IPython.display import display, Image as IPImage


### All of the remaining code in this notebook is solely for using the
### built-in Overcooked state visualizer on a trained agent, so that you can see
### a graphical rendering of what your agents are doing. It is not
### necessary to use this.

# The below code is a particular way to rollout episodes in a format
# compatible with the built-in state visualizer.

class StudentPolicy(NNPolicy):
    """ Generate policy """
    def __init__(self, env, agent):
        super(StudentPolicy, self).__init__()
        self.env = env
        self.agent = agent

    def state_policy(self, state, agent_index):
        """
        This method should be used to generate the poiicy vector corresponding to
        the state and agent_index provided as input.  If you're using a neural
        network-based solution, the specifics depend on the algorithm you are using.
        Below are two commented examples, the first for a policy gradient algorithm
        and the second for a value-based algorithm.  In policy gradient algorithms,
        the neural networks output a policy directly.  In value-based algorithms,
        the policy must be derived from the Q value outputs of the networks.  The
        uncommented code below is a placeholder that generates a random policy.
        """
        featurized_state = self.env.base_env.featurize_state_mdp(state)
        input_state = torch.FloatTensor(featurized_state[agent_index]).unsqueeze(0)

        # Example for policy NNs named "PNN0" and "PNN1"
        with torch.no_grad():
            if agent_index == 0:
                action_probs = self.agent.actor0(input_state)[0].numpy()
            else:
                action_probs = self.agent.actor1(input_state)[0].numpy()

        return action_probs

    def multi_state_policy(self, states, agent_indices):
        """ Generate a policy for a list of states and agent indices """
        return [self.state_policy(state, agent_index) for state, agent_index in zip(states, agent_indices)]


class StudentAgent(AgentFromPolicy):
    """Create an agent using the policy created by the class above"""
    def __init__(self, policy):
        super(StudentAgent, self).__init__(policy)


