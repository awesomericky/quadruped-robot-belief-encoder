import argparse
from ruamel.yaml import YAML
import numpy as np
import torch

from agent import VisionStudentAgent


class Environment:
    """
    Change to the environment you are using
    """
    def __init__(self, args):
        self.obs_dim = args.proprio_obs_dim + args.extero_obs_dim
        self.action_dim = args.action_dim
        self.n_envs = args.n_envs

    def observe(self):
        observations = np.random.normal(size=(self.n_envs, self.obs_dim)).astype(np.float32)
        return observations

    def observe_noisy(self):
        noisy_observations = np.random.normal(size=(self.n_envs, self.obs_dim)).astype(np.float32)
        return noisy_observations

    def step(self, action):
        rewards = np.random.normal(size=self.n_envs).astype(np.float32)
        dones = np.zeros(shape=self.n_envs).astype(np.bool_)
        return rewards, dones

class TeacherAgent:
    """
    Change to the teacher agent you are using
    """
    def __init__(self, args):
        self.action_dim = args.action_dim
        self.n_envs = args.n_envs

    def getAction(self, observations):
        actions = np.random.normal(size=(self.n_envs, self.action_dim)).astype(np.float32)
        return actions

def getParser():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--name', type=str, default='example')
    parser.add_argument('--device', type=str, default='cuda', help='gpu or cpu.')
    parser.add_argument('--save_dir', type=str, default='example', help='directory name to save weights')
    return parser

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    # parameters to be set from the environment you are running
    args.student_model_num = 0
    args.student_lr = 3e-4
    args.student_epochs = 1
    args.max_grad_norm = 1.
    args.student_policy_type = "vision_recurrent"
    args.n_envs = 100
    args.n_steps = 40000
    args.n_steps_per_env = int(args.n_steps / args.n_envs)

    args.proprio_obs_dim = 10 #100
    args.extero_obs_dim = 20 #200
    args.action_dim = 12

    # config
    cfg = YAML().load(open("config.yaml", 'r'))

    # define teacher agent (pretrained)
    teacher = TeacherAgent(args)

    # define student agent
    student = VisionStudentAgent(args, cfg["student_model"])
    hidden_state_tensor = None

    # define environment
    env = Environment(args)

    max_update = 10

    for update in range(max_update):
        for _ in range(args.n_steps_per_env):
            obs = env.observe()
            noisy_obs = env.observe_noisy()

            proprio_obs = obs[:, :args.proprio_obs_dim]
            extero_obs = obs[:, -args.extero_obs_dim:]
            noisy_extero_obs = noisy_obs[:, -args.extero_obs_dim:]
            proprio_obs_tensor = torch.from_numpy(proprio_obs).to(args.device)
            noisy_extero_obs_tensor = torch.from_numpy(noisy_extero_obs).to(args.device)

            with torch.no_grad():
                # get student action
                actions_tensor, hidden_state_tensor = student.getAction(
                    proprio_state=proprio_obs_tensor,
                    extero_state=noisy_extero_obs_tensor,
                    hidden_state=hidden_state_tensor
                )

                # get teacher action
                teacher_actions = teacher.getAction(obs)

            actions = actions_tensor.detach().cpu().numpy()
            rewards, dones = env.step(actions)

            # add data the buffer
            student.step(proprio_obs, noisy_extero_obs, extero_obs, teacher_actions)

        # train model
        loss, reconstruction_loss, action_loss = student.train()

        # save model
        if update % 5 == 0:
            student.save(update)

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("total loss: ", '{:6.4f}'.format(loss)))
        print('{:<40} {:>6}'.format("reconstruction loss: ", '{:6.4f}'.format(reconstruction_loss)))
        print('{:<40} {:>6}'.format("action loss: ", '{:6.4f}'.format(action_loss)))
        print('----------------------------------------------------\n')

