import numpy as np
import torch

class VisionRolloutBuffer:
    def __init__(self, args):
        self.proprio_dim = args.proprio_obs_dim
        self.extero_dim = args.extero_obs_dim
        self.action_dim = args.action_dim
        self.n_envs = args.n_envs
        self.n_steps = args.n_steps
        self.n_steps_per_env = int(self.n_steps/self.n_envs)
        self.device = args.device

        self.cnt = 0
        self.proprio_states = np.zeros((self.n_steps_per_env, self.n_envs, self.proprio_dim), dtype=np.float32)
        self.noisy_extero_states = np.zeros((self.n_steps_per_env, self.n_envs, self.extero_dim), dtype=np.float32)
        self.extero_states = np.zeros((self.n_steps_per_env, self.n_envs, self.extero_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps_per_env, self.n_envs, self.action_dim), dtype=np.float32)

    def addTransition(self, proprio_states, noisy_extero_states, extero_states, actions):
        """
        :param proprio_states: proprioceptive sensor data [numpy.float32]
        :param noisy_extero_states: noisy exteroceptive sensor data [numpy.float32]
        :param extero_states: (teacher) exteroceptive sensor data [numpy.float32]
        :param actions: (teacher) action [numpy.float32]
        :return:
        """
        assert self.cnt < self.n_steps_per_env
        self.proprio_states[self.cnt] = proprio_states
        self.noisy_extero_states[self.cnt] = noisy_extero_states
        self.extero_states[self.cnt] = extero_states
        self.actions[self.cnt] = actions
        self.cnt += 1

    def getBatches(self):
        """
        :return: (L, N, D)
        """
        self.cnt = 0
        proprio_states_tensor = torch.from_numpy(self.proprio_states).to(self.device)
        noisy_extero_states_tensor = torch.from_numpy(self.noisy_extero_states).to(self.device)
        extero_states_tensor = torch.from_numpy(self.extero_states).to(self.device)
        actions_tensor = torch.from_numpy(self.actions).to(self.device)
        return proprio_states_tensor, noisy_extero_states_tensor, extero_states_tensor, actions_tensor
