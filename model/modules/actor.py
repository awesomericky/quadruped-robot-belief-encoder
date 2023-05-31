import torch.nn as nn
import torch

from base_nn import BaseNet


class RecurrentAttentionPolicy(BaseNet):
    class BeliefEncoder(BaseNet):
        def __init__(self, model_cfg):
            super().__init__(model_config=model_cfg)

        def forward(self, proprio_state, encoded_extero_state, hidden_state=None):
            """
            :param proprio_state: proprioceptive sensor data [(L, H) / (L. N, H)]
            :param encoded_extero_state: encoded exteroceptive sensor data [(L, H) / (L. N, H)]
            :param hidden_state: hidden state of the recurrent layer [(N_layer, H) / (N_layer, N, H)]
            :return:
                recurrent_output: output state of the recurrent layer [(L, H) / (L. N, H)]
                recurrent_hidden: next hidden state of the recurrent layer [(N_layer, H) / (N_layer, N, H)]
                belief_state: next belief state of the belief encoder [(L, H) / (L. N, H)]
            """
            output = dict()

            fused_state = torch.cat((proprio_state, encoded_extero_state), dim=-1)  #[(L, H) / (L. N, H)]
            if hidden_state is None:
                output["recurrent_output"], output["recurrent_hidden"] = self.recurrent_encoder(fused_state)  # (L. N, H)
            else:
                output["recurrent_output"], output["recurrent_hidden"] = self.recurrent_encoder(fused_state, hidden_state)  # (L. N, H)

            tensor_shape = torch.Size(torch.ones_like(torch.tensor(output["recurrent_output"].shape)[:-1])) + (4,)
            output["belief_state"] = \
                torch.tile(self.state_encoder(output["recurrent_output"]), dims=tensor_shape) + \
                nn.functional.sigmoid(self.attention_encoder(output["recurrent_output"])) * encoded_extero_state
            return output

    class BeliefDecoder(BaseNet):
        def __int__(self, model_cfg):
            super().__int__(model_config=model_cfg)

        def forward(self, extero_state, hidden_state):
            """
            :param extero_state: exteroceptive sensor data [(L, H) / (L. N, H)]
            :param hidden_state: output state of the recurrent layer [(L, H) / (L. N, H)]
            (cf: In GRU, last output state is same as the hidden state)
            :return:
                estimated_extero_state: estimated exteroceptive sensor data [(L, H) / (L. N, H)]
            """
            estimated_extero_state = \
                self.extero_decoder(hidden_state) + \
                nn.functional.sigmoid(self.attention_encoder(hidden_state)) * extero_state
            return estimated_extero_state

    def __init__(self, args, model_cfg):
        self.proprio_dim = args.proprio_obs_dim
        self.extero_dim = args.extero_obs_dim
        self.action_dim = args.action_dim

        self.device = args.device
        self.args = args
        self.model_cfg = model_cfg
        self.adapt_model()

        super(RecurrentAttentionPolicy, self).__init__(model_config=model_cfg["policy"])
        self.belief_encoder = self.BeliefEncoder(model_cfg["belief_encoder"])
        self.belief_decoder = self.BeliefDecoder(model_cfg["belief_decoder"])

    def adapt_model(self):
        assert self.extero_dim % 4 == 0
        self.model_cfg["policy"]["MLP"]["extero_encoder"]["input"] = self.extero_dim // 4
        self.model_cfg["policy"]["MLP"]["base_net"]["input"] = \
            self.proprio_dim + self.model_cfg["policy"]["MLP"]["extero_encoder"]["output"] * 4
        self.model_cfg["policy"]["MLP"]["base_net"]["output"] = self.action_dim

        self.model_cfg["belief_encoder"]["GRU"]["recurrent_encoder"]["input"] = \
            self.proprio_dim + self.model_cfg["policy"]["MLP"]["extero_encoder"]["output"] * 4

        self.model_cfg["belief_encoder"]["MLP"]["attention_encoder"]["input"] \
            = self.model_cfg["belief_encoder"]["GRU"]["recurrent_encoder"]["hidden"]
        self.model_cfg["belief_encoder"]["MLP"]["attention_encoder"]["output"] \
            = self.model_cfg["policy"]["MLP"]["extero_encoder"]["output"] * 4
        self.model_cfg["belief_encoder"]["MLP"]["state_encoder"]["input"] \
            = self.model_cfg["belief_encoder"]["GRU"]["recurrent_encoder"]["hidden"]

        self.model_cfg["belief_decoder"]["MLP"]["attention_encoder"]["input"] \
            = self.model_cfg["belief_encoder"]["GRU"]["recurrent_encoder"]["hidden"]
        self.model_cfg["belief_decoder"]["MLP"]["attention_encoder"]["output"] \
            = self.extero_dim
        self.model_cfg["belief_decoder"]["MLP"]["extero_decoder"]["input"] \
            = self.model_cfg["belief_encoder"]["GRU"]["recurrent_encoder"]["hidden"]
        self.model_cfg["belief_decoder"]["MLP"]["extero_decoder"]["output"] \
            = self.extero_dim

    def forward(self, proprio_state, extero_state, hidden_state=None, use_decoder=True):
        """
        :param proprio_state: proprioceptive sensor data [(L, H) / (L. N, H)]
        :param extero_state: exteroceptive sensor data [(L, H) / (L. N, H)]
        :param hidden_state: hidden state of the recurrent layer in the belief encoder [(N_layer, H) / (N_layer, N, H)]
        :param use_decoder: use belief decoder to estimate exteroceptive data or not
        :return:
            action: [(L, H) / (L. N, H)]
            recurrent_hidden: [(N_layer, H) / (N_layer, N, H)]
            estimated_extero_state: estimated exteroceptive sensor data [(L, H) / (L. N, H)]
        """
        output = dict()

        length_and_batch = proprio_state.shape[:-1]
        encoded_extero_state = extero_state.view(*length_and_batch, 4, int(self.extero_dim / 4))
        encoded_extero_state = self.extero_encoder(encoded_extero_state).view(*length_and_batch, -1)

        belief_encoder_output = self.belief_encoder(proprio_state, encoded_extero_state, hidden_state)
        fused_state = torch.cat((proprio_state, belief_encoder_output["belief_state"]), dim=-1)
        output["action"] = self.base_net(fused_state)
        output["recurrent_hidden"] = belief_encoder_output["recurrent_hidden"]

        if use_decoder:
            output["estimated_extero_state"] = \
                self.belief_decoder(extero_state, belief_encoder_output["recurrent_output"])
        return output

    def getAction(self, proprio_state, extero_state, hidden_state):
        """
        :param proprio_state: [(N, H)]
        :param extero_state: [(N, H)]
        :param hidden_state: [(N_layer, N, H)]
        :return:
            action: [(N, H)]
            hidden_state: [(N_layer, N, H)]
        """
        assert len(proprio_state.shape) == 2 and proprio_state.shape[0] == self.args.n_envs
        proprio_state = proprio_state.unsqueeze(0)
        extero_state = extero_state.unsqueeze(0)
        output = self.forward(proprio_state, extero_state, hidden_state, use_decoder=False)
        action = output["action"].squeeze(0)
        next_hidden_state = output["recurrent_hidden"].squeeze(0)
        return action, next_hidden_state