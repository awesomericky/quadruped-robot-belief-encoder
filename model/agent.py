from torch.optim import Adam
import torch
import os

from modules.actor import RecurrentAttentionPolicy
from storage import VisionRolloutBuffer
from utils.color import cprint

policy_modules = {"vision_recurrent": RecurrentAttentionPolicy}
storage_modules = {"vision_recurrent": VisionRolloutBuffer}


class VisionStudentAgent:
    def __init__(self, args, model_cfg):
        # base
        self.device = args.device
        self.name = args.name
        self.model_num = args.student_model_num
        self.checkpoint_dir = f'{args.save_dir}/student_checkpoint'

        # for regression
        self.student_lr = args.student_lr
        self.student_epochs = args.student_epochs
        self.max_grad_norm = args.max_grad_norm

        # for models
        assert args.student_policy_type in policy_modules.keys()
        Policy = policy_modules[args.student_policy_type]
        self.actor = Policy(args, model_cfg).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.student_lr)

        # for data storage
        Storage = storage_modules[args.student_policy_type]
        self.rollout_buffer = Storage(args)

        self.epoch = self.load()

    def getAction(self, proprio_state, extero_state, hidden_state):
        """
        :param proprio_state: [(N, H)]
        :param extero_state: [(N, H)]
        :param hidden_state: [(N_layer, N, H)]
        :return:
            action: [(N, H)]
            next_hidden_state: [(N_layer, N, H)]
        """
        return self.actor.getAction(proprio_state, extero_state, hidden_state)

    def step(self, proprio_state, noisy_extero_state, extero_state, action):
        self.rollout_buffer.addTransition(proprio_state, noisy_extero_state, extero_state, action)

    def train(self):
        proprio_states_tensor, noisy_extero_states_tensor, extero_states_tensor, actions_tensor \
            = self.rollout_buffer.getBatches()

        total_loss = 0
        total_reconstruction_loss = 0
        total_action_loss = 0

        for _ in range(self.student_epochs):
            # forward pass
            output = self.actor(proprio_states_tensor, noisy_extero_states_tensor)
            student_action = output["action"]
            estimated_extero_state = output["estimated_extero_state"]

            # compute loss
            reconstruction_loss = torch.mean(torch.pow(estimated_extero_state - extero_states_tensor, 2))
            action_loss = torch.mean(torch.pow(student_action - actions_tensor, 2))
            loss = 0.5 * reconstruction_loss + action_loss

            # optimize
            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # logging
            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_action_loss += action_loss.item()

        total_loss /= self.student_epochs
        total_reconstruction_loss /= self.student_epochs
        total_action_loss /= self.student_epochs

        return total_loss, total_reconstruction_loss, total_action_loss

    def load_exteroceptive_encoder(self, checkpoint):
        """
        :param checkpoint: teacher model torch checkpoint (dict)  (cf: Only "actor"!)
        """
        loaded_checkpoint = dict()
        for k, v in checkpoint.items():
            if k.split('.')[0] == "extero_encoder":
                loaded_checkpoint['.'.join(k.split('.')[1:])] = v

        if len(loaded_checkpoint.keys()) != 0:
            self.actor.extero_encoder.load_state_dict(loaded_checkpoint)
            cprint("Exteroceptive encoder load success", bold=True, color="blue")
        else:
            cprint("Exteroceptive encoder load fail", bold=True, color="blue")

    def save(self, model_name):
        save_dict = {
            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict()
        }
        torch.save(save_dict, f"{self.checkpoint_dir}/full_{model_name}.pt")
        cprint(f'[{self.name} - full_{model_name}.pt] save success.', bold=True, color="blue")

    def load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/full_{self.model_num}.pt"

        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            cprint(f'[{self.name} - full_{self.model_num}.pt] load success.', bold=True, color="blue")
            return int(self.model_num)
        else:
            cprint(f'[{self.name} - full_{self.model_num}.pt] load fail.', bold=True, color="red")
            return 0
