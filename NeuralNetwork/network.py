# %%
import torch
import torch.nn as nn
# %%
class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(NeuralNetwork, self).__init__()
        HIDDEN_SIZE = 16
        self.layer_in = nn.Linear(input_size, HIDDEN_SIZE)
        self.layer_hidden1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer_hidden2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer_hidden3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer_out  = nn.Linear(HIDDEN_SIZE, output_size)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer_in(x))
        x = self.relu(self.layer_hidden1(x))
        x = self.relu(self.layer_hidden2(x))
        x = self.relu(self.layer_hidden3(x))
        x = self.layer_out(x)
        return x