import torch

class FCQ(torch.nn.Module):
    def __init__(self, states_input_size, actions_size, 
                hidden_layers,
                activation_fn=torch.nn.functional.relu,
                optimizer=torch.optim.Adam, learning_rate=0.0005) -> None:
        super().__init__()
        self.activation_fn = activation_fn

        self.hidden_layers = torch.nn.ModuleList()
        prev_size = states_input_size
        for layer_size in hidden_layers:
            self.hidden_layers.append(torch.nn.Linear(prev_size, layer_size))
            prev_size = layer_size

        self.output_layer = torch.nn.Linear(prev_size, actions_size)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)


    def format_(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        return states


    def forward(self, states):
        x = self.format_(states)
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        return self.output_layer(x)


    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    @staticmethod
    def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def reset(self):
        self.apply(FCQ.reset_weights)