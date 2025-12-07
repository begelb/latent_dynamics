''' This file adapted from MORALS'''


from torch import nn

class Encoder(nn.Module):
    def __init__(self, config):
        num_layers = config.num_layers
        hidden_shape = config.hidden_shape
        input_shape = config.high_dims
        lower_shape = config.low_dims
        
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.encoder.add_module(f"linear_{i}", nn.Linear(input_shape, hidden_shape))
            else:
                self.encoder.add_module(f"linear_{i}", nn.Linear(hidden_shape, hidden_shape))
            self.encoder.add_module(f"relu_{i}", nn.ReLU(True))
        self.encoder.add_module(f"linear_{num_layers}", nn.Linear(hidden_shape, lower_shape))
        self.encoder.add_module(f"tanh_{num_layers}", nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        num_layers = config.num_layers
        hidden_shape = config.hidden_shape
        input_shape = config.high_dims
        lower_shape = config.low_dims

        super(Decoder, self).__init__()

        self.decoder = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.decoder.add_module(f"linear_{i}", nn.Linear(lower_shape, hidden_shape))
            else:
                self.decoder.add_module(f"linear_{i}", nn.Linear(hidden_shape, hidden_shape))
            self.decoder.add_module(f"relu_{i}", nn.ReLU(True))
        self.decoder.add_module(f"linear_{num_layers}", nn.Linear(hidden_shape, input_shape))
        self.decoder.add_module(f"sigmoid_{num_layers}", nn.Sigmoid())

    def forward(self, x):
        x = self.decoder(x)
        return x

class LatentDynamics(nn.Module):
    # Takes as input an encoding and returns a latent dynamics
    # vector which is just another encoding
    def __init__(self, config):
        num_layers = config.num_layers
        hidden_shape = config.hidden_shape
        lower_shape = config.low_dims

        super(LatentDynamics, self).__init__()

        self.dynamics = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.dynamics.add_module(f"linear_{i}", nn.Linear(lower_shape, hidden_shape))
            else:
                self.dynamics.add_module(f"linear_{i}", nn.Linear(hidden_shape, hidden_shape))
            self.dynamics.add_module(f"relu_{i}", nn.ReLU(True))
        self.dynamics.add_module(f"linear_{num_layers}", nn.Linear(hidden_shape, lower_shape))
        self.dynamics.add_module(f"tanh_{num_layers}", nn.Tanh())
    
    def forward(self, x):
        x = self.dynamics(x)
        return x
