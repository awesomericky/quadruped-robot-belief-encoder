import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, model_config):
        super(BaseNet, self).__init__()
        self.model_config = model_config
        self.activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}

        used_model_archs = self.model_config.keys()

        if "LSTM" in used_model_archs:
            for model_name, model_arch in self.model_config["LSTM"].items():
                self.add_module(model_name, nn.LSTM(
                    input_size=model_arch["input"],
                    hidden_size=model_arch["hidden"],
                    num_layers=model_arch["num_layers"],
                    batch_first=model_arch["batch_first"],
                    dropout=model_arch["dropout"]
                ))

        if "GRU" in used_model_archs:
            for model_name, model_arch in self.model_config["GRU"].items():
                self.add_module(model_name, nn.GRU(
                    input_size=model_arch["input"],
                    hidden_size=model_arch["hidden"],
                    num_layers=model_arch["num_layers"],
                    batch_first=model_arch["batch_first"],
                    dropout=model_arch["dropout"]
                ))

        if "MLP" in used_model_archs:
            for model_name, model_arch in self.model_config["MLP"].items():
                assert model_arch["activation"] in list(self.activation_map.keys()), "Unavailable activation."
                self.add_module(model_name, MLP(
                    input_size=model_arch["input"],
                    output_size=model_arch["output"],
                    shape=model_arch["shape"],
                    activation=self.activation_map[model_arch["activation"]],
                    dropout=model_arch["dropout"] if "dropout" in model_arch.keys() else 0.,
                    batchnorm=model_arch["batchnorm"] if "batchnorm" in model_arch.keys() else False
                ))

        if "Linear" in used_model_archs:
            for model_name, model_arch in self.model_config["Linear"].items():
                self.add_module(model_name, nn.Linear(
                    in_features=model_arch["input"],
                    out_features=model_arch["output"]
                ))


class MLP(nn.Module):
    def __init__(self, input_size, output_size, shape, activation, dropout=0.0, batchnorm=False):
        super(MLP, self).__init__()
        self.activation_fn = activation

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            if batchnorm:
                modules.append(nn.BatchNorm1d(shape[idx+1]))
            modules.append(self.activation_fn())
            if dropout != 0.0:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, input):
        return self.architecture(input)